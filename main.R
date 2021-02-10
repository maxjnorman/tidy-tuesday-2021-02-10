# https://www.tidymodels.org/start/case-study/
library(tidyverse)
library(tidymodels)
library(glmnet)
library(vip)

# scaling functions
scale <- list()
scale[["penalty"]] <- function(val) {val * (10^-1 - 10^-4) + 10^-4}

# static vector
holidays <- c("AllSouls", "AshWednesday", "ChristmasEve", "Easter", 
              "ChristmasDay", "GoodFriday", "NewYearsDay", "PalmSunday")

loc_dat <- "https://tidymodels.org/start/case-study/hotels.csv"
hotels <- read_csv(loc_dat) %>%
  mutate_if(is.character, as.factor) 
dim(hotels)

set.seed(123)
# partition into training and testing data
splits <- hotels %>%
  initial_split(prop = (4/5), strata = children)
test <- splits %>%
  testing()
train <- splits %>%
  training()
val <- train %>%
  validation_split(prop = (3/4))

# proportion of children overall
hotels %>% 
  count(children) %>% 
  mutate(prop = n/sum(n))
# training set proportions by children
train %>%
  count(children) %>%
  mutate(prop = n/sum(n))
# test set proportions by children
test %>%
  count(children) %>%
  mutate(prop = n/sum(n))

# start building the model + gubbins
logit <- list() # container for logit model
logit[["model"]] <- logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet") %>%
  set_mode("classification")
logit[["grid"]] <- tibble(penalty = runif(150) %>% scale$penalty())
logit[["recipe"]] <- recipe(children ~ ., data = train) %>% 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date, holidays = holidays) %>% 
  step_rm(arrival_date) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())
logit[["workflow"]] <- workflow() %>% 
  add_model(logit[["model"]]) %>% 
  add_recipe(logit[["recipe"]])
logit[["result"]] <- logit[["workflow"]] %>%
  tune_grid(
    val,
    grid = logit[["grid"]],
    control = control_grid(save_pred = TRUE),
    metrics = metric_set(roc_auc)
  )
logit[["result"]] %>% show_best("roc_auc", 3)
# plot the result of the search
logit[["plot"]] <- logit[["result"]] %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())
logit[["plot"]] 

tree <- list()
tree[["model"]] <- decision_tree(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet") %>%
  set_mode("classification")
