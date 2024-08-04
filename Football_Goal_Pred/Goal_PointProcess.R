require(dplyr)
require(tidyr)
require(ggplot2)
require(hrbrthemes)
require(wesanderson)



home_goal_times <- vector("list", nrow(combinedData))

for (i in 1:nrow(combinedData)) {
  
  goal_times <- c()
  
  
  for (j in 1:15) {
    time_col <- paste0("GOAL_", j, "_TIME")
    team_col <- paste0("GOAL_", j, "_TEAM")
    
    
    if (!is.na(combinedData[i, team_col]) && combinedData[i, team_col] == "home") {
      goal_times <- c(goal_times, combinedData[i, time_col])
    }
  }
  
  # If no home goal times were found, add 0 (as numeric) to the vector
  if (length(goal_times) == 0) {
    #  goal_times <- as.integer(0)
    #  goal_times <- NA
      goal_times <- numeric(0)
  }
  
  home_goal_times[[i]] <- goal_times
}


away_goal_times <- vector("list", nrow(combinedData))


for (i in 1:nrow(combinedData)) {
  
  goal_times <- c()
  
  
  for (j in 1:15) {
    time_col <- paste0("GOAL_", j, "_TIME")
    team_col <- paste0("GOAL_", j, "_TEAM")
    
    
    if (!is.na(combinedData[i, team_col]) && combinedData[i, team_col] == "away") {
      goal_times <- c(goal_times, combinedData[i, time_col])
    }
  }
  
  # If no away goal times were found, add 0 (as num) to the vector
  if (length(goal_times) == 0) {
    #  goal_times <- as.integer(0)
    #  goal_times <- NA
      goal_times <- numeric(0)
  }
  
  # Store the vector in the list
  away_goal_times[[i]] <- goal_times
}


homeGoalTimes <- home_goal_times
awayGoalTimes <- away_goal_times

allGoalTimes <- c(homeGoalTimes, awayGoalTimes)

homeProbsStrengths <- combinedData$B365H
awayProbsStrengths <- combinedData$B365A

allStrengths <- c(homeProbsStrengths, awayProbsStrengths)

hist(unlist(allGoalTimes), breaks = seq(0, 90, by = 1), right = FALSE, 
     main = "Histogram of List Entries",
     xlab = "Time/min", ylab = "Frequency", col = "lightblue", border = "black")

diract <- function(t, x=90){
  2*as.numeric((round(t) == x))
}

qplot(seq(0, 100, 0.1), diract(seq(0, 100, 0.1))) + 
  xlab("Time") + 
  ylab("Weight")

intensityFunction <- function(params, t, winProb, maxT){
  beta0 <- params[1]
  beta1 <- params[2]
  beta90 <- params[3]
  
  int <- (winProb * beta0) + (beta1 * (t/maxT)) + (beta90*diract(t))
  int[int < 0] <- 0
  int
}

intensitFunctionInt <- function(params, maxT, winProb){
  beta0 <- params[1]
  beta1 <- params[2]
  beta90 <- params[3]
  
  beta0*winProb*maxT + (beta1*maxT)/2 + beta90
}

likelihood <- function(params, t, winProb){
  ss <- sum(log(intensityFunction(params, t, winProb, 90)))
  int <- intensitFunctionInt(params, 90, winProb)
  ss - int
}

sim_events <- function(params, winProb){
  lambdaMax <- 1.1*intensityFunction(params, 90, winProb, 90)
  nevents <- rpois(1, lambdaMax*90)
  tstar <- runif(nevents, 0, 90)
  accept_prob <- intensityFunction(params, tstar, winProb, 90) / lambdaMax
  (sort(tstar[runif(length(accept_prob)) < accept_prob]))
}

N <- 100
testParams <- c(3, 2, 2)
testWinProb <- 1

testEvents <- replicate(N, sim_events(testParams, testWinProb))
testWinProbs <- rep_len(testWinProb, N)

trueInt <- intensityFunction(testParams, 0:90, testWinProb, 90)

alllikelihood <- function(params, events, winProbs){
  ll <- sum(vapply(seq_along(events), 
                   function(i) likelihood(params, events[[i]], winProbs[[i]]), 
                   numeric(1)))
  if(ll == -Inf){
    return(-1e9)
  } else {
    return(ll)
  }
}

trueLikelihood <- alllikelihood(testParams, testEvents, testWinProbs)

simRes <- optim(runif(3), function(x) -1*alllikelihood(c(x[1], x[2], x[3]), 
                                                       testEvents, 
                                                       testWinProbs), lower = c(0,0,0), method = "L-BFGS-B")

print(simRes$par)


simResDF <- data.frame(Time = 0:90, 
                       TrueIntensity = trueInt, 
                       EstimatedIntensity = intensityFunction(simRes$par, 0:90, testWinProb, 90))

ggplot(simResDF, aes(x=Time, y=TrueIntensity, color = "True")) + 
  geom_line() + 
  geom_line(aes(y=EstimatedIntensity, color = "Estimated")) + 
  labs(color = NULL) + 
  xlab("Time") + 
  ylab("Intensity") + 
  theme(legend.position = "bottom")



trainInds <- sample.int(length(allGoalTimes), size = floor(length(allGoalTimes)*0.7))

goalTimesTrain <- allGoalTimes[trainInds]
strengthTrain <- allStrengths[trainInds]

goalTimesTest <- allGoalTimes[-trainInds]
strengthTest <- allStrengths[-trainInds]

optNull <- optim(runif(1), function(x) -1*alllikelihood(c(x[1], 0, 0), 
                                                        goalTimesTrain, 
                                                        strengthTrain), lower = c(0,0,0), method = "L-BFGS-B")
optNull

optNull2 <- optim(runif(2), function(x) -1*alllikelihood(c(x[1], x[2], 0), 
                                                         goalTimesTrain, 
                                                         strengthTrain), lower = c(0,0,0), method = "L-BFGS-B")
optNull2

optRes <- optim(runif(3), function(x) -1*alllikelihood(x, 
                                                       goalTimesTrain, 
                                                       strengthTrain), lower = c(0,0,0), method = "L-BFGS-B")
optRes

optRes2 <- optim(runif(2), function(x) -1*alllikelihood(c(x[1], 0, x[2]), 
                                                        goalTimesTrain, 
                                                        strengthTrain), lower = c(0,0,0), method = "L-BFGS-B")
optRes2

modelFits <- data.frame(Time = 0:90)
modelFits$Null <- intensityFunction(c(optNull$par[1],0,0), modelFits$Time, 2, 90)
modelFits$Linear <- intensityFunction(c(optNull2$par ,0), modelFits$Time, 2, 90)
modelFits$Delta <- intensityFunction(optRes$par, modelFits$Time, 2, 90)
modelFits$NoLinear <- intensityFunction(c(optRes2$par[1], 0, optRes2$par[2]), modelFits$Time, 2, 90)

modelFits %>% 
  pivot_longer(!Time, names_to="Model", values_to="Intensity") -> modelFitsTidy

ggplot(modelFitsTidy, aes(x=Time, y=Intensity, color = Model)) + 
  geom_line() + 
  theme(legend.position = "bottom")

maxGoals <- vapply(strengthTest, 
                   function(x) max(replicate(100, length(sim_events(optRes$par, x)))),
                   numeric(1))

actualMaxGoals <- max(vapply(allGoalTimes, length, numeric(1)))

ggplot(data = data.frame(MaxGoals = maxGoals), aes(x=MaxGoals)) + 
  geom_histogram(binwidth = 1) + 
  geom_vline(xintercept = actualMaxGoals) + 
  xlab("Maximum Number of Goals")
