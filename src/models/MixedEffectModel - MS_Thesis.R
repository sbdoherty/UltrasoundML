
#######################################################
# Load appropriate libraries 
#######################################################

# install.packages("Rcpp", type = "source")
# install.packages("RcppEigen", type = "source")
# install.packages("lme4", type = "source")
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(multcomp)
library(rms)
library(nlme)
library(ggplot2)
library(viridis)
library(Rcpp)

theme_set(theme_bw())

setwd('C:\\Users\\doherts\\Documents\\SVN\\MULTIS_Studies\\LayerEffectOnStiffness')
#######################################################
# Functions
#######################################################

run_model_noRand <- function(equation, ex_data){
  ML.model <- lm(equation, data=ex_data)
  return(ML.model)
}

run_model <- function(equation, ex_data){
  ML.model <- lmer(equation, data=ex_data)
  return(ML.model)
}

#######################################################
# Load data
#######################################################
# Read in data 
train.data <- read.csv("dat/train_data.csv", header=TRUE, sep=",", na.strings="NA", dec=".")
test.data <- read.csv("dat/test_data.csv", header=TRUE, sep=",", na.strings="NA", dec=".")

print('Intercept locations combined percent diff:')
print(min(train.data$Compliance))
print(max(train.data$Compliance))
print(mean(train.data$Compliance))

#######################################################
# Intercept Models
#######################################################
# Intercept model fit to all locations together 
M1 = run_model(Compliance ~ Muscle + Fat + (1|SubID) + (1|Location), train.data)
M1_test = predict(M1, newdata = test.data)

print(summary(M1))
print(M1_test)
test.data$Prediction <- M1_test

percent_diff <- abs(test.data$Prediction-test.data$Compliance)/(test.data$Compliance)*100
test.data$PercDiff <- percent_diff
print('Intercept locations combined percent diff:')
print(mean(test.data$PercDiff))
print(sd(test.data$PercDiff))

ggplot(test.data, aes(x=Prediction, y=Compliance, color=Location)) + geom_point(alpha=.5, size=1)+
  theme(text = element_text(size=6), legend.key.size = unit(0.3, "cm"))+
  labs(x=bquote("Predicted compliance" ~(mm^3/N)), y=bquote("Experimental compliance" ~(mm^3/N)))+
  guides(colour = guide_legend(override.aes = list(size=2)))+
  geom_abline(slope=1, intercept=0, linetype='dashed', size=1)+
  xlim(range(c(test.data$Compliance, test.data$Prediction))) + 
  ylim(range(c(test.data$Compliance, test.data$Prediction)))
ggsave('doc/Figures/ms_thesis--StatsModel_actual_predicted_ALLTogether.png', height = 5, width = 8, units='cm')

# Location specific statistical model 
by_loc <- train.data %>% group_by(Location)
by_loc_test <- test.data %>% group_by(Location)
my_split <- group_split(by_loc_test)
i = 0 
total <- NA
for (val in group_split(by_loc))
{
  i = i + 1
  print(first(val['Location'][[1]]))
  temp_df <- as.data.frame(my_split[i])[[1]][[1]]
  print(temp_df)
  M1 = run_model_noRand(Compliance ~ Muscle + Fat , val)
  
  # model_p <- M1[['model']]
  M1_test = predict(M1, newdata = temp_df)
  model_p <- head(M1[['model']],length(M1_test))
  model_p['Fitted'] <- M1_test
  model_p["InvStiff"] <- temp_df$Compliance
  model_p['Location'] <- temp_df['Location']
  model_p['Fat'] <- temp_df['Fat']
  model_p['Muscle'] <- temp_df['Muscle']
  
  print(model_p['Location'][1])
  if (model_p['Location'][1] == 'LA_A'){
    total <- model_p
  } else{
    total <- rbind(total, model_p)
  }
  print(summary(M1))
}

percent_diff <- abs(total['Fitted']-total['InvStiff'])/(total['InvStiff'])*100
print('Intercept locations specific percent diff:')
print(mean(percent_diff$Fitted))
print(sd(percent_diff$Fitted))

ggplot(total, aes(x=Fitted, y=InvStiff, color=Location)) + geom_point(alpha=.5, size=1)+
  theme(text = element_text(size=6), legend.key.size = unit(0.3, "cm"))+
  labs(x=bquote("Predicted compliance" ~(mm^3/N)), y=bquote("Experimental compliance" ~(mm^3/N)))+
  guides(colour = guide_legend(override.aes = list(size=2)))+
  geom_abline(slope=1, intercept=0, linetype='dashed', size=1)+
  xlim(range(c(total$Fitted, total$InvStiff))) + 
  ylim(range(c(total$Fitted, total$InvStiff)))
ggsave('doc/Figures/MS_thesis--StatsModel_actual_predicted.png', height = 5, width = 8, units='cm')

ggplot(total, aes(x=Muscle, y=InvStiff, color=Location))+ geom_point(alpha=.5, size=1)+
  theme(text = element_text(size=6), legend.key.size = unit(0.3, "cm"))+
  guides(colour = guide_legend(override.aes = list(size=2)))+
  labs(x=bquote("Muscle Thickness (mm)"), y=bquote("Experimental compliance" ~(mm^3/N)))
ggsave('doc/Figures/ms_thesis--Thick_compliance_muscle.png', height = 5, width = 8, units='cm')

ggplot(total, aes(x=Fat, y=InvStiff, color=Location))+ geom_point(alpha=.5, size=1)+
  theme(text = element_text(size=6), legend.key.size = unit(0.3, "cm"))+
  guides(colour = guide_legend(override.aes = list(size=2)))+
  labs(x=bquote("Fat Thickness (mm)"), y=bquote("Experimental compliance" ~(mm^3/N)))
ggsave('doc/Figures/ms_thesis--Thick_compliance_fat.png', height = 5, width = 8, units='cm')


#######################################################
# Physics based models
#######################################################
# Physics based model fit to all locations together 
M1 = run_model(Compliance ~ Muscle + Fat + (1|SubID) + (1|Location)-1, train.data)
print(summary(M1))
M1_test = predict(M1, newdata = test.data)

print(summary(M1))
print(M1_test)
test.data$PhysPred <- M1_test
percent_diff <- abs(test.data$PhysPred-test.data$Compliance)/(test.data$Compliance)*100
test.data$PercDiff <- percent_diff
print('Physics ALL locations combined percent diff:')
print(mean(test.data$PercDiff))
print(sd(test.data$PercDiff))

ggplot(test.data, aes(x=PhysPred, y=Compliance, color=Location)) + geom_point(alpha=.5, size=1)+
  theme(text = element_text(size=6), legend.key.size = unit(0.3, "cm"))+
  labs(x=bquote("Predicted compliance" ~(mm^3/N)), y=bquote("Experimental compliance" ~(mm^3/N)))+
  guides(colour = guide_legend(override.aes = list(size=2)))+
  geom_abline(slope=1, intercept=0, linetype='dashed', size=1)+
  xlim(range(c(test.data$Compliance, test.data$PhysPred))) + 
  ylim(range(c(test.data$Compliance, test.data$PhysPred)))
ggsave('doc/Figures/ms_thesis--PhysicsModel_actual_predicted_ALLTogether.png', height = 5, width = 8, units='cm')

# Location specific physics based model 

# Location specific statistical model 
by_loc <- train.data %>% group_by(Location)
by_loc_test <- test.data %>% group_by(Location)
my_split <- group_split(by_loc_test)
i = 0 
total <- NA
for (val in group_split(by_loc))
{
  i = i + 1
  print(first(val['Location'][[1]]))
  temp_df <- as.data.frame(my_split[i])[[1]][[1]]
  M1 = run_model_noRand(Compliance ~ Muscle + Fat -1, val)
  
  # model_p <- M1[['model']]
  M1_test = predict(M1, newdata = temp_df)
  model_p <- head(M1[['model']],length(M1_test))
  model_p['Fitted'] <- M1_test
  model_p["InvStiff"] <- temp_df$Compliance
  model_p['Location'] <- temp_df['Location']
  model_p['Fat'] <- temp_df['Fat']
  model_p['Muscle'] <- temp_df['Muscle']
  
  if (model_p['Location'][1] == 'LA_A'){
    total <- model_p
  } else{
    total <- rbind(total, model_p)
  }
  print(summary(M1))
}

percent_diff <- abs(total['Fitted']-total['InvStiff'])/(total['InvStiff'])*100
print('Physics-based locations specific percent diff:')
print(mean(percent_diff$Fitted))
print(sd(percent_diff$Fitted))

ggplot(total, aes(x=Fitted, y=InvStiff, color=Location)) + geom_point(alpha=.5, size=1)+
  theme(text = element_text(size=6), legend.key.size = unit(0.3, "cm"))+
  labs(x=bquote("Predicted compliance" ~(mm^3/N)), y=bquote("Experimental compliance" ~(mm^3/N)))+
  guides(colour = guide_legend(override.aes = list(size=2)))+
  geom_abline(slope=1, intercept=0, linetype='dashed', size=1)+
  xlim(range(c(total$Fitted, total$InvStiff))) + 
  ylim(range(c(total$Fitted, total$InvStiff)))
ggsave('doc/Figures/ms_thesis--PhysicsModel_actual_predicted.png', height = 5, width = 8, units='cm')

