require(readr)
require(dplyr)
require(tidyr)
require(ggplot2)
require(rstanarm)
require(mgcv)
require(caret)
require(lubridate)
require(patchwork)
require(memoise)
require(rvest)

get_fandom_epl_football_data <- function(season = "2018-19", matchweek = 1) {
  match_url <- glue::glue("https://football.fandom.com/wiki/{season}_Premier_League:_Match_day_{matchweek}")
  
  raw_list <- 
    rvest::read_html(match_url) %>% 
    rvest::html_table()
  
  games <- raw_list[2:11] %>% 
    dplyr::bind_rows() %>% 
    dplyr::filter(
      !is.na(X2), 
      stringr::str_detect(X3, pattern = "Report", negate = TRUE)
    ) %>% 
    dplyr::bind_cols(
      raw_list[2:11] %>% 
        dplyr::bind_rows() %>% 
        dplyr::filter(
          !is.na(X2), 
          stringr::str_detect(X3, pattern = "Report")
        ) %>% 
        dplyr::select(X6 = X2, X7 = X4)
    ) %>% 
    tidyr::separate(
      col = "X3", 
      into = c("home_score", "away_score"), 
      sep = "-"
    ) %>% 
    dplyr::mutate(
      home_score = as.numeric(home_score), 
      away_score = as.numeric(away_score)
    ) %>% 
    dplyr::rename(
      datetime = X1, 
      home = X2, 
      away = X4, 
      match_info = X5, 
      home_comments = X6, 
      away_comments = X7
    )
  
  return(games)
}

variabledude <- get_fandom_epl_football_data()
