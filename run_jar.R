setwd("Documents/repos/dfg_2018_hiwi")
library(rJava)

.jinit()
.jaddClassPath(path = "/home/flo/Documents/repos/dfg_2018_hiwi/TimeSeriesClassification.jar")
trainAndPredict = .jnew("myimpl.TrainAndPredict")

train_data = "GunPoint/GunPoint_TRAIN.arff"
test_data = "GunPoint/GunPoint_TEST.arff"
model = "GunPoint/myobj.txt"
classifName = "weka.classifiers.meta.RotationForest"
hyperpars = ""
args_train   = c(train_data, model, classifName, hyperpars)
args_predict = c(model, test_data)

mod = J(trainAndPredict, "train", args_train)

J(trainAndPredict, "predict", args_predict)







