
##load libraries
library(readr)
library(tidyverse)
library(corrr)
library(caret)
library(glmnet)
library(randomForest)
library(plyr)
library(dplyr)
library(purrr)
library(modelr)

#########################################################################
# NOTE: Since the steps 1 through 4 are the same for all three periods
# (2006 to 2009, 2010 to 2012, and 2013 to 2016), these steps have been
# shown only for period 1 (2006-2009). The exact same process can
# be followed for periods 2 and 3.
#########################################################################

# Importing data, and limiting it to the data to until 2016
data <- read_csv("~/input.filepath/input.filename")
data <- filter(data, year != 2017)

# Slicing the data into three parts based on prescription overdose phase
# Assigning a logical variable "highPrescribe" to each phase:
# 1 if prescription per 100 persons is greater than the median and 0 otherwise

# Data for phase 1 (2006 to 2009)
p1_data <- data %>% 
  filter(year %in% 2006:2009)
quantile(p1_data$PR_per_100) #108.2
p1_data$highPrescribe <- ifelse(p1_data$PR_per_100 >= 108.2, 1, 0)


###############################################################
# PERIOD 1, Prescription opioid phase, 2006-2009
###############################################################

# Obtaining the correlation values for the first period
# Focusing on correlations of highPrescribe (response) with all other variables
set.seed(123)
correlations <- p1_data %>% 
  select(-c(county, state_abbrev, PR_per_100, year, name)) %>% 
  correlate() %>%
  focus(highPrescribe) %>%        
  as.data.frame()

# Removing those variables with "NA" correlation to response (highPrescribe)
remove <- correlations %>% 
  filter(is.na(highPrescribe))

remove_rows = remove[['rowname']] %>% 
  as.vector()
keep = setdiff(colnames(p1_data), remove_rows)
p1_data = p1_data[keep]

## Removing variables that are highly correlated (0.8 cutoff), by type of exposure (PMC, PDMP, IRX)
# This will help with Lasso accuracy in the next step
# PMC (Pain Management Programs)
correlations = select(p1_data, PMC_Law:PMC_Inspect) %>% 
  cor()
highlyCorrelated = findCorrelation(correlations, cutoff=(0.8),verbose = FALSE)
important_varPMC = colnames(correlations[,-highlyCorrelated]) %>% 
  as.data.frame()    

# PDMP (prescription drug monitoring programs)
correlations = select(p1_data, agency_Law.enforcement_PAdmin:share.provision_None.of.the.above.restrictions_pdmpR) %>% 
  cor()
highlyCorrelated = findCorrelation(correlations, cutoff=(0.8),verbose = FALSE)
important_varPDMP = colnames(correlations[,-highlyCorrelated]) %>% 
  as.data.frame()

# IRX (initial prescribing limits)
correlations = select(p1_data, IRX_Initial_RX_Law:IRX_Subs_Spec_Sched2) %>%
 cor()
highlyCorrelated = findCorrelation(correlations, cutoff=(0.8),verbose = FALSE)
important_varIRX = colnames(correlations[,-highlyCorrelated]) %>%
 as.data.frame()

# Binding the remaining variables for each type of exposure
# These form the data that will be used for the subsequent Lasso
p1_imp_var = bind_rows(important_varPMC, important_varPDMP) %>% 
  as_vector()
p1_imp_subset = p1_data[p1_imp_var] %>%  as.data.frame()
response = p1_data$highPrescribe %>%  as.data.frame()
p1_lasso = bind_cols(response, p1_imp_subset)                   
names(p1_lasso)[1] = "highPrescribe"                          

###########LASSO#################

# Setting up the model matrix and specifying the outcome
x = model.matrix(highPrescribe~., p1_lasso)[,-1]
y = p1_lasso$highPrescribe

# Running Lasso regression with a binomial specification and finding coefficients
set.seed(123)
cvfit = cv.glmnet(x, y, nfolds = 10, alpha = 1, family = "binomial", type.measure="auc") 
finalLasso = glmnet(x, y, nfolds = 10, alpha = 1, lambda = cvfit$lambda.1se, family = "binomial", type.measure="auc")
coefs = coef(finalLasso)

# Selecting those variables with non-zero coefficients which will be used 
# for the subsequent Random Forest
nonzero_coefs = which((coefs[,1]) != 0)
p1_rf = p1_lasso[nonzero_coefs]                                 

# Factorizing the response
p1_rf = p1_rf %>%                                             
  mutate(
    highPrescribe = as.factor(highPrescribe)
  )
# Saving data used to generate random forest to a csv file
write_csv(p1_rf, "p1_rf.csv")


#############################################################################################
# Code and annotations for permutation analysis has been developed by Dr. Jeanette Stingone #
# Assistant Professor of Epidemiology, Mailman School pf Public Health, Columbia University #
# https://www.publichealth.columbia.edu/people/our-faculty/js5406                           #
#############################################################################################

###############################################################
# RANDOM FOREST
###############################################################

# It would be helpful to input study-specific parameters in the first step
input.filepath<-"~/input.filepath"
base.filepath<-"~/base.filepath"
## Filename of data used to generate random forest
input.filename<-"p1_rf.csv"
## Response or dependent variable
outcome_var<-"highPrescribe"
## Number of trees to be grown in the forest
num_trees<-500
## Number of variables randomly selected for inclusion at each node
varnum<-15
## Minimum node size within the forest
nodesz<-100
## Factor level of outcome that is of primary interest (1 for high prescribe)
pred.select<-"1"
## Number of permutations of outcome variable to generate null combinations
num_perm<-1000
## Maximum number of terminal nodes within a tree
max_nodes<-10
## The selected threshold to identify combinations that are incompatible with the null distribution
threshold.select<-"threshold_95"

dataset<-read.csv(file=paste0(input.filepath, input.filename), header=TRUE)

# Factorizing the outcome
dataset[[outcome_var]]<-as.factor(dataset[[outcome_var]])

# Running Random Forest to find most important provisions
formula_rf<-as.formula(paste(outcome_var, '~ .', sep=""))
rf.1<-randomForest(formula=formula_rf, data=dataset, ntree=num_trees, mtry=varnum, nodesize=nodesz, maxnodes=max_nodes )


###############################################################
# PERMUTATION ANALYSIS
###############################################################

# Obtaining paths from trees within forest
## Returning the rules of a tree

###############################################################
# Functions reference:
# https://stats.stackexchange.com/questions/41443/how-to-actually-plot-a-sample-tree-from-randomforestgettree
getConds<-function(tree){
  #store all conditions into a list
  conds<-list()
  #start by the terminal nodes and find previous conditions
  id.leafs<-which(tree$status==-1)
  j<-0
  for(i in id.leafs){
    j<-j+1
    prevConds<-prevCond(tree,i)
    conds[[j]]<-prevConds$cond
    while(prevConds$id>1){
      prevConds<-prevCond(tree,prevConds$id)
      conds[[j]]<-paste(conds[[j]]," & ",prevConds$cond)
    }
    if(prevConds$id==1){
      conds[[j]]<-paste(conds[[j]]," => ",tree$prediction[i])
    }
  }
  
  return(conds)
}

# Finding the previous conditions in the tree
prevCond<-function(tree,i){
  if(i %in% tree$right_daughter){
    id<-which(tree$right_daughter==i)
    cond<-paste(tree$split_var[id],">",tree$split_point[id])
  }
  if(i %in% tree$left_daughter){
    id<-which(tree$left_daughter==i)
    cond<-paste(tree$split_var[id],"<",tree$split_point[id])
  }
  
  return(list(cond=cond,id=id))
}

#remove spaces in a word
collapse<-function(x){
  x<-sub(" ","_",x)
  
  return(x)
}
###############################################################

num_paths<-vector()
num_nodes<-vector()
root.node.var<-vector()
root.node.split<-vector()
forest.path<-list()
forest.pred<-list()
forest.val<-list()

#Get information on a tree
for (n in 1:num_trees){
  
  tree<-getTree(rf.1, n, labelVar = TRUE)
  #rename the name of the column to remove any spaces
  colnames(tree)<-sapply(colnames(tree),collapse)
  rules<-getConds(tree)
  
  #Obtain count of paths within a tree
  tree$status<-as.factor(tree$status)
  counts<-summary(tree$status)
  num_paths[[n]]<-counts[[1]]
  #Obtain number of total nodes in tree
  num_nodes[[n]]<-nrow(tree)
  
  #Obtain info for root node
  tree$split_var<-as.character(tree$split_var)
  tree$status<-as.numeric(tree$status)
  root.node.var[[n]]<-tree$split_var[1]
  root.node.split[[n]]<-tree$split_point[1]
  
  #Decompose trees and store branch informatoin
  tree.path.vars<-list()
  tree.path.pred.class<-list()
  tree.path.vals<-list()
  
  #Loop goes through each branch within a tree
  for (i in 1:length(rules)){
    
    rule.temp<-rules[[i]]
    rule.temp.parts<-unlist(strsplit(rule.temp, " "))
    tree.path.pred.class[[i]]<-rule.temp.parts[length(rule.temp.parts)]
    splits<-round(length(rule.temp.parts)/6)
    
    split.vars<-vector()
    split.vals<-vector()
    
    #Loop goes through all splits within a branch
    for (j in 1:splits) {
      
      split.vars[[j]]<-paste(rule.temp.parts[(length(rule.temp.parts)-6*j)],
                             (rule.temp.parts[(length(rule.temp.parts)-6*j)+1]))
      split.vals[[j]]<-paste(rule.temp.parts[length(rule.temp.parts)-6*j+2])
    } 
    tree.path.vars[[i]]<-paste(split.vars, sep='', collapse = '')
    tree.path.vals[[i]]<-paste(split.vals, collapse=" / ")
  }
  forest.path[[n]]<-tree.path.vars
  forest.pred[[n]]<-tree.path.pred.class
  forest.val[[n]]<-tree.path.vals
}


## Identifying paths associated with predicted outcome of interest
paths <- unlist(forest.path)
preds <- unlist(forest.pred)
vals <- unlist(forest.val)

paths.preds <- cbind(paths, preds)
colnames(paths.preds) <- c("path", "prediction")

paths.preds.select <- paths.preds[which(paths.preds[,2]==pred.select),]


## Determining frequency of paths and root node and output files
# Creating count of full paths
counts<-table(paths.preds.select)

# Creating count of root nodes across trees
count.rn<-table(root.node.var)

prep.file<-as.data.frame(counts)
prep.file2<-as.data.frame(count.rn)

prep.file.a<-prep.file[order(-prep.file$Freq),]

filepath.1<-paste(base.filepath,"\\output_count_of_paths.csv", sep="")
filepath.2<-paste(base.filepath,"\\output_count_of_rootnodes.csv", sep="")

write.csv(prep.file.a, file=filepath.1)
write.csv(prep.file2, file=filepath.2)

## Determining Thresholds from Null Analysis
# Permute Outcome Variable and Run Forests to Create Count of Null Combinations
permute_dataset<-permute(dataset, num_perm, outcome_var)

rand_forests<-map(permute_dataset$perm, ~ randomForest(
  formula=formula_rf,data=., ntree=num_trees, mtry=varnum,
  nodesize=nodesz, maxnodes=max_nodes))

num_paths.rand<-vector()
num_nodes.rand<-vector()
root.node.var.rand<-vector()
root.node.split.rand<-vector()
forest.path.rand<-list()
forest.pred.rand<-list()
prep.file.rand<-list()
counts.rand<-list()

# Get information on a tree
for (m in 1:num_perm){
  
  rf.p.1<-rand_forests[[m]]
  
  for (n in 1:num_trees){
    
    tree<-getTree(rf.p.1, n, labelVar = TRUE)
    
    #rename the name of the column to remove any spaces
    colnames(tree)<-sapply(colnames(tree),collapse)
    rules<-getConds(tree)
    
    #Obtain count of paths within a tree
    tree$status<-as.factor(tree$status)
    counts.rand.tree<-summary(tree$status)
    num_paths.rand[[n]]<-counts.rand.tree[[1]]
    
    #Obtain number of total nodes in tree
    num_nodes.rand[[n]]<-nrow(tree)
    
    #Obtain info for root node
    tree$split_var<-as.character(tree$split_var)
    tree$status<-as.numeric(tree$status)
    root.node.var.rand[[n]]<-tree$split_var[1]
    root.node.split.rand[[n]]<-tree$split_point[1]
    
    tree.path.vars.rand<-list()
    tree.path.pred.class.rand<-list()
    
    #Loop through all branches within a tree
    
    for (i in 1:length(rules)){
      
      rule.temp<-rules[[i]]
      rule.temp.parts<-unlist(strsplit(rule.temp, " "))
      tree.path.pred.class.rand[[i]]<-rule.temp.parts[length(rule.temp.parts)]
      splits.rand<-round(length(rule.temp.parts)/6)
      
      split.vars.rand<-vector()
      
      #Loop through all splits within a branch
      
      for (j in 1:splits.rand) {
        
        split.vars.rand[[j]]<-paste(rule.temp.parts[(length(rule.temp.parts)-6*j)],
                                    (rule.temp.parts[(length(rule.temp.parts)-6*j)+1]))
      } 
      tree.path.vars.rand[[i]]<-paste(split.vars.rand, sep='', collapse = '')
    }
    forest.path.rand[[n]]<-tree.path.vars.rand
    forest.pred.rand[[n]]<-tree.path.pred.class.rand
  }
  
  paths.rand<-unlist(forest.path.rand)
  preds.rand<-unlist(forest.pred.rand)
  
  paths.preds.rand<-cbind(paths.rand,preds.rand)
  colnames(paths.preds.rand)<-c("path","prediction")
  
  paths.preds.select.rand<-paths.preds.rand[which(paths.preds.rand[,2]==pred.select),]
  
  counts.rand[[m]]<-table(paths.preds.select.rand)
  prep.file.rand[[m]] <-as.data.frame(counts.rand[[m]])
  
}

comb.list.r<-ldply(prep.file.rand, rbind)

comb.list.r.final<-filter(comb.list.r, comb.list.r$paths.preds.select.rand != pred.select)

# Thresholds provided for the 95%, 99% and 99.5% of the distribution across forests
threshold_995<-quantile(comb.list.r.final$Freq, probs=c(0.995))
threshold_99<-quantile(comb.list.r.final$Freq, probs=c(0.99))
threshold_95<-quantile(comb.list.r.final$Freq, probs=c(0.95))

rand.results<-cbind(threshold_95, threshold_99, threshold_995)

filepath.3<-paste(base.filepath,"\\thresholds_randomtrees.csv", sep="")

write.csv(rand.results, file=filepath.3)

## Storing values of threshold splits
threshold.select.2 <- get(threshold.select)
# Limiting to branches with frequencies above the threshold
paths.preds.pass <- as.data.frame(prep.file.a[which(prep.file.a$Freq > threshold.select.2),])
paths.preds < -as.data.frame(paths.preds)
# Finding the original indices of those branches
paths.preds.2 <- tibble::rownames_to_column(paths.preds, "INDEX")
colnames(paths.preds.pass)<-c("path", "values")

# Using those indices to retrieve the values used by the trees during recursive partitioning
paths.values <- inner_join(paths.preds.2, paths.preds.pass, by="path")
vals <- as.data.frame(vals)
vals.2 <- tibble::rownames_to_column(vals, "INDEX")

paths.num.values<-inner_join(paths.values, vals.2, by="INDEX")