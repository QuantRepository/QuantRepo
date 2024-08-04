require(readr)
require(dplyr)
require(tidyr)
require(ggplot2)
require(rstanarm)
require(mgcv)
require(caret)
require(lubridate)
require(patchwork)


fn <- list.files("YOUR_PATH", full.names = T)
rawDataList <- lapply(fn, read_csv, guess_max=10000)
rawData <- bind_rows(rawDataList)
rm(rawDataList)


rawData %>% 
  select(Div, Date, HomeTeam, AwayTeam, 
         FTHG, FTAG, HTHG, HTAG,
         PSCH, PSCA, PSCD) %>% 
  drop_na -> cleanData

rawData %>% 
  select(Div, Date, HomeTeam, AwayTeam,
         PSCH, PSCA, PSCD, PSH, PSA, PSD, B365H, B365A, B365D) %>% 
  drop_na -> cleanData

rawData %>% 
  select(Div, Date, HomeTeam, AwayTeam,
         FTHG, FTAG, HTHG, HTAG,
         B365H, B365A, B365D) %>% 
  drop_na -> cleanData

cleanData %>% 
  select(Div, Date, HomeTeam, FTHG, HTHG, PSCH, PSCA, PSCD) %>% 
  rename(Team = HomeTeam, FTG = FTHG, HTG = HTHG, 
         WinOdds = PSCH, LoseOdds = PSCA, DrawOdds = PSCD) %>% 
  mutate(Home = 1) -> homeData

cleanData %>% 
  select(Div, Date, AwayTeam, FTAG, HTAG, PSCH, PSCA, PSCD) %>% 
  rename(Team = AwayTeam, FTG = FTAG, HTG = HTAG, 
         WinOdds = PSCA, LoseOdds = PSCH, DrawOdds = PSCD) %>% 
  mutate(Home = 0) -> awayData

allData <- bind_rows(homeData, awayData)
allData %>% 
  mutate(WinProbRaw = 1/WinOdds,
         LoseProbRaw = 1/LoseOdds, 
         DrawProbRaw = 1/DrawOdds,
         WinProb = WinProbRaw /(WinProbRaw + LoseProbRaw + DrawProbRaw),
         LoseProb = LoseProbRaw /(WinProbRaw + LoseProbRaw + DrawProbRaw),
         DrawProb = DrawProbRaw /(WinProbRaw + LoseProbRaw + DrawProbRaw),
         Date = dmy(Date)) -> allData

trainInds <- createDataPartition(allData$Div, p = 0.7, list=FALSE)

trainData <- allData[trainInds, ]
testData <- allData[-trainInds, ]

nullModel <- glm(FTG ~ 1, data=trainData, family="poisson")

winOddsModel <- glm(FTG ~ WinProb, family = "poisson", data=trainData)

winOddsPolyModel <- glm(FTG ~ WinProb + 
                          I(WinProb^2) + 
                          I(WinProb^3) + 
                          I(WinProb^4), family = "poisson", 
                        data=trainData)

winOddsGAMModel <- gam(FTG ~ s(WinProb), family="poisson", 
                       data=trainData)

oddsGrid <- data.frame(WinProb=seq(0, 1, by=0.01))
oddsShape <- predict(winOddsModel, newdata = oddsGrid)
polyShape <- predict(winOddsPolyModel, newdata = oddsGrid)
gamShape <- predict(winOddsGAMModel, newdata = oddsGrid)
nullShape <- predict(nullModel, newdata = oddsGrid)

oddsGrid %>% 
  mutate(Linear = oddsShape, 
         Polynomial = polyShape, 
         GAM = gamShape,
         Null = nullShape) -> oddsGrid

oddsGrid %>% gather(Model, Value, -WinProb) -> oddsGridTidy

testData %>% 
  group_by(WinProbBucket = cut(WinProb, breaks = seq(0, 1, by=0.01))) %>% 
  summarise(N=n(),
            ActualGoals = mean(FTG),
            AvgWinProb = mean(WinProb)) %>% 
  ungroup %>% mutate(Model = "Emperical") -> sumData

ggplot() + 
  geom_line(data = oddsGridTidy, aes(x=WinProb, y=exp(Value), colour=Model)) +
  geom_point(data=sumData, aes(x=AvgWinProb, y=ActualGoals, colour=Model, alpha=scale(N)), size=2, show.legend = F) + 
  theme(legend.position = "bottom", legend.title = element_blank()) + 
  ylab("Number of Goals") + 
  xlab("Probability of Winning")

modelList <- list(NullModel = nullModel,
                  WinOdds = winOddsModel,
                  WinOddsPoly = winOddsPolyModel,
                  WinOddsGAM = winOddsGAMModel)

modelNames <- c("Null", "Linear", "Poly", "GAM")

lambdas <- lapply(modelList, 
                  function(x) exp(predict(x, newdata=testData)))
logLikelihoodTest <- lapply(lambdas, 
                            function(x) sum(dpois(testData$FTG, x, log = T)))

logLikelihoodTestFrame <- data.frame(Model = modelNames, 
                                     LogLikelihoods = unlist(logLikelihoodTest), 
                                     Parameters = nrow(trainData) - vapply(modelList, df.residual, numeric(1)))

logLikelihoodTestFrame %>% 
  arrange(-LogLikelihoods)

bayesModel <- stan_glm(FTG ~ WinProb + 
                         I(WinProb^2) + 
                         I(WinProb^3) + 
                         I(WinProb^4), 
                       family = "poisson", 
                       data=trainData, 
                       chains=2)

oddsGridBayes <- data.frame(WinProb = seq(0, 1, by=0.01))

linpreds <- posterior_linpred(bayesModel, newdata=oddsGridBayes)

as.data.frame(t(linpreds)) %>% 
  mutate(WinProb = oddsGridBayes$WinProb) %>% 
  gather(Iteration, Value, -WinProb) %>%
  mutate(ValueExp = exp(Value)) -> linpredsPlot

ggplot(linpredsPlot, aes(x=WinProb, y=ValueExp)) + 
  stat_summary(fun.y=mean, geom="line", colour="blue") + 
  stat_summary(fun.ymin = function(x) quantile(x, 0.05), fun.ymax = function(x) quantile(x, 0.95), geom="ribbon", alpha=0.5) + 
  geom_point(data=sumData, aes(x=AvgWinProb, y=ActualGoals, colour=Model, alpha=scale(N)), size=2, show.legend = F) + 
  xlab("Winning Probability") +
  ylab("Number of Goals")

prop_zero <- function(x) mean(x == 0)

maxPPC <- pp_check(bayesModel, plotfun = "ppc_stat", stat="max", binwidth=1) 
zeroPropPPC <- pp_check(bayesModel, plotfun = "ppc_stat", stat = "prop_zero", binwidth = 0.01) 

maxPPC + zeroPropPPC


nbModel <- stan_glm(FTG ~ WinProb + 
                      I(WinProb^2) + 
                      I(WinProb^3) + 
                      I(WinProb^4), 
                    family = neg_binomial_2, 
                    data=trainData, 
                    chains=2, 
                    cores = 2, 
                    iter= 1000 )
maxPPC_nb <- pp_check(nbModel, plotfun = "ppc_stat", stat="max", binwidth=1)
zeroPropPPC_nb<- pp_check(nbModel, plotfun = "ppc_stat", stat = "prop_zero", 
                          binwidth = 0.01) 

maxPPC_nb + zeroPropPPC_nb


standardize_date <- function(date_str) {
  # Detect the date format based on string length
  if (nchar(date_str) == 10) {
    date_obj <- as.Date(date_str, format = "%d/%m/%Y")
  } else if (nchar(date_str) == 8) {
    date_obj <- as.Date(date_str, format = "%d/%m/%y")
  } else {
    stop("Unrecognized date format")
  }
  
  # Format date to "01/01/00"
  formatted_date <- format(date_obj, format = "%d/%m/%y")
}
cleanData$Date <- sapply(cleanData$Date, standardize_date)
