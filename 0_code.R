# You can download my data and code
# https://github.com/karesma0/Heavy-rain-damage-prediction-function-1.git

Train_D <- read.csv("1_training set.csv")
Test_D <- read.csv("2_test set.csv")

library(rpart)  
library(partykit) 
library(randomForest)
library(caret)
library(e1071)

NRMSE <- function(yi, yhat_i){
  (sqrt(mean((yi - yhat_i)^2)))/(max(yi)-min(yi))
} 

for(j in 1:7) {
  for(i in j:7) {
    
    Train_1 <- Train_D[, c(1, (2+27*(j-1)):(1+27*i))]
    
    ## Linear Regression model(with PCA)
    Train_sub <- Train_1[,-c(1)] 
    preProc <- preProcess(Train_sub, method = c("center", "scale", "zv", "nzv", "pca"), thresh = 0.9)
    PC_train <- predict(preProc, newdata = Train_sub)
    Damage <- Train_1[,1]
    Train_2 <- cbind.data.frame(Damage, PC_train)
    
    LR_train <- lm(Damage~., Train_2)
    LR_model <- step(LR_train, direction = "both")
    
    
    ## Decision Tree model
    set.seed(180926)
    tree <- rpart(Damage ~., data=Train_1)
    min_cp <- tree$cptable[which.min(tree$cptable[,'xerror']),]
    Tree_model <- prune(tree, cp = min_cp[1]) 
    
    
    ## Random Forest model
    set.seed(180926)
    RF <- randomForest(Damage ~., data = Train_1)
    RF_model <- randomForest(Damage ~., data = Train_1, ntree = which.min(RF$mse))
    
    
    ## Support Vector Machine model
    svm.radial <- tune(svm, Damage ~., data = Train_1, kernel = "radial",
                       ranges = list(cost = 2^(0:7), epsilon = seq(0,1,0.1)))
    SVM_model <- svm.radial$best.model
    
    
    ## Predict heavy rainfall damage
    Val_1 <- Test_D[, c(1, (2+27*(j-1)):(1+27*i))]
    PC_val <- predict(preProc, newdata = Val_1) 
    
    sim_LR <- predict(LR_model, newdata = PC_val)
    sim_Tree <- predict(Tree_model, newdata = Val_1)
    sim_RF <- predict(RF_model, newdata = Val_1) 
    sim_SVM <- predict(SVM_model, newdata = Val_1) 
    
    
    ## Reverse Log Transform
    real_sim_LR <- exp(sim_LR)
    real_sim_Tree <- exp(sim_Tree)
    real_sim_RF <- exp(sim_RF) 
    real_sim_SVM <- exp(sim_SVM)
    
    
    ## Evaluate data
    NRMSE_LR <- NRMSE(exp(Val_1$Damage), real_sim_LR)
    NRMSE_Tree <- NRMSE(exp(Val_1$Damage), real_sim_Tree)
    NRMSE_RF <- NRMSE(exp(Val_1$Damage), real_sim_RF)  
    NRMSE_SVM <- NRMSE(exp(Val_1$Damage), real_sim_SVM)
    
    
    ## Print result
    Result <- cbind(NRMSE_LR, NRMSE_Tree, NRMSE_RF, NRMSE_SVM)
    
    Name <- c("1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", 
              "2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", 
              "3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", 
              "4-1", "4-2", "4-3", "4-4", "4-5", "4-6", "4-7", 
              "5-1", "5-2", "5-3", "5-4", "5-5", "5-6", "5-7", 
              "6-1", "6-2", "6-3", "6-4", "6-5", "6-6", "6-7", 
              "7-1", "7-2", "7-3", "7-4", "7-5", "7-6", "7-7") 
    hoho <- paste0('output/')
    hoho[i+7*(j-1)] <- paste0('output/',Name[i+7*(j-1)],'.csv')
    write.csv(Result, hoho[i+7*(j-1)])
  }
}
