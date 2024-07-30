
# config ------------------------------------------------------------------


library(tidyverse)
library(mlr3verse)
library(mlr3proba)
library(transGI)
library(future.apply)

data_all = read_csv('data/flowchart.csv')[, -1]

docal = function(x_, fun_, ...) {
  
  params_ = lapply(as.list(match.call())[-c(1:3)], \(x__) eval(x__, envir = x_))
  if (is.null(names(params_)) | '' %in% names(params_)) names(params_) = names(formals(fun_))
  res_ = do.call(fun_, params_)
  
  return(res_)
  
}

caler = function(fea_, fun_, onError_, ...) {
  
  # browser()
  which_ = which(!is.na(fea_))
  params_ = list(...)
  params_[['fea_']] = fea_
  params_ = lapply(params_, \(x__) x__[which_])
  res_ = tryCatch(
    do.call(fun_, params_), 
    error = function(e) onError_
  )
  
  return(res_)
  
}

num_na = data_all |> 
  is.na() |> 
  colSums()
to_drop = names(which(num_na>4823))

data_pre = data_all |> 
  select(!all_of(to_drop)) |> 
  filter(!is.na(OS) & !is.na(OS.time)) |> 
  filter(OS.time < 0) |> 
  rename(OS_time = OS.time) |> 
  mutate(across(where(is.character), as.factor), 
         across(where(~ is.numeric(.x) && length(unique(.x, na.rm = T)) < 6), as.integer), 
         across(where(~ is.factor(.x) && length(unique(.x, na.rm = T)) < 6), as.integer))


plan('multisession', workers = 4)
# uni_cindex = future_lapply(data_pre[, -c(1:2)], caler, fun_ = cal_cindex, onError_ = 0.5, time_ = data_pre$OS_time, event_ = data_pre$OS)
uni_HR = future_lapply(data_pre[, -c(1:2)], caler, fun_ = cal_HR, onError_ = 1, time_ = data_pre$OS_time, event_ = data_pre$OS)
plan('sequential')

toKeep = uni_HR |> 
  bind_rows(.id = 'fea') |> 
  filter(p < 0.05) |> 
  filter(abs(HR-1)>0.5) |> 
  pull(fea); toKeep = c('OS', 'OS_time', toKeep)

# build -------------------------------------------------------------------

## uni-selection ---

uni_cindex = data_pre |> 
  mutate(across(-c(1, 2), ~ cal_cindex(.x, time_ = OS_time, event_ = OS)))

## tasks ----

split = sample(1:nrow(data_pre), size = round(0.75*nrow(data_pre))); split = list(train = split, test = setdiff(1:nrow(data_pre), split))
task_train = as_task_surv(data_pre[split$train, toKeep], time = 'OS_time', event = 'OS')
task_test = as_task_surv(data_pre[split$test, toKeep],  time = 'OS_time', event = 'OS')

## process ----

preprocess_pipe0 = 
  po("imputelearner", id = "int", lrn("classif.lightgbm"), affect_columns = selector_type("integer")) %>>%
  po("imputelearner", id = "num", lrn("regr.lightgbm"), affect_columns = selector_type("numeric"))

preprocess_pipe0$train(task_train)

task_train_ready = preprocess_pipe0$predict(task_train)[[1]]
task_test_ready  = preprocess_pipe0$predict(task_test)[[1]]

## predict ---- 

lrn_base_xgb = lrn('surv.xgboost')

lrn_base_xgb$train(task_train_ready)

pre_base_xgb_train = lrn_base_xgb$predict(task_train_ready)
pre_base_xgb_test  = lrn_base_xgb$predict(task_test_ready)

pre_base_xgb_train$score()
pre_base_xgb_test$score()







