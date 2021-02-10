# https://www.tidymodels.org/start/case-study/
library(tidyverse)
library(tidymodels)
library(glmnet)
library(vip)
library(parallel)

all_cores <- detectCores(logical = FALSE)

# scaling functions
standardise_factory <- function(bounds = c(0, 1)) {
  standardise_instance <- function(val) {
    val * (max(bounds) - min(bounds)) + min(bounds)
  }
  return(standardise_instance)
}
scale <- list()
scale[["penalty"]] <- standardise_factory(bounds = c(10^-1, 10^-4))
scale[["tree_depth"]] <- standardise_factory(bounds = c(1, 16))
scale[["cost_complexity"]] <- standardise_factory()

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
tree[["model"]] <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune()
) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")
tree[["grid"]] <- tibble(
  cost_complexity = runif(150) %>% scale$cost_complexity(),
  tree_depth = runif(150) %>% scale$tree_depth()
)
tree[["recipe"]] <- recipe(children ~ ., data = train) %>% 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date, holidays = holidays) %>% 
  step_rm(arrival_date) %>% 
  step_dummy(all_nominal(), -all_outcomes())
tree[["workflow"]] <- workflow() %>% 
  add_model(tree[["model"]]) %>% 
  add_recipe(tree[["recipe"]])
tree[["result"]] <- tree[["workflow"]] %>%
  tune_grid(
    val,
    grid = tree[["grid"]],
    control = control_grid(save_pred = TRUE),
    metrics = metric_set(roc_auc)
  )
tree[["result"]] %>% show_best("roc_auc", 3)
tree[["plot-tree_depth"]] <- tree[["result"]] %>% 
  collect_metrics() %>% 
  ggplot(aes(x = cost_complexity, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())
tree[["plot-tree_depth"]]
tree[["plot-cost_complexity"]] <- tree[["result"]] %>% 
  collect_metrics() %>% 
  ggplot(aes(x = tree_depth, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())
tree[["plot-cost_complexity"]]