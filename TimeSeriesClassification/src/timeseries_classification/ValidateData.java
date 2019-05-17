package timeseries_classification;
import utilities.ClassifierTools;
import weka.core.AttributeStats;
import weka.core.Instances;

public class ValidateData {
    public boolean validation(Instances data){
        // check whether data has class column
        String name =data.attribute(data.classIndex()).name();
        boolean errFlag = true;
        if (!name.equals("class")&&!name.equals("traget")){
            System.err.println("Warning! the data don't have class column");
            errFlag = false;

        }
        for (int i = 0; i < data.numAttributes();i++){
            AttributeStats as = data.attributeStats(i);
            float percent = Math.round(100.0 * as.missingCount / as.totalCount);
            if(percent>50){
                System.err.println(String.format("Warning! The data missing percentage of column %d is %f percent", i+1,percent));
                errFlag = false;
            }
        }

        if (errFlag == false){
            System.exit(0);
        }
        return errFlag;
    }

//    public static void main(String[] args) {
////        String dataAddress = "/Users/wangyu/Downloads/datasets/bin_train";
////        String dataAddress = "/Users/wangyu/Downloads/datasets/bin_test";
////        String dataAddress = "/Users/wangyu/Downloads/datasets/bin_train_nocl";
////        String dataAddress = "/Users/wangyu/Downloads/datasets/bin_test_nocl";
////        String dataAddress = "/Users/wangyu/Downloads/datasets/bin_test_nacl";
////        String dataAddress = "/Users/wangyu/Downloads/datasets/full_na_lab_train";
////        String dataAddress = "/Users/wangyu/Downloads/datasets/full_na_train";
////        String dataAddress = "/Users/wangyu/Downloads/datasets/part_na_train";
////        String dataAddress = "/Users/wangyu/Downloads/datasets/full_na_test";
////        String dataAddress = "/Users/wangyu/Downloads/datasets/mult_test";
////        String dataAddress = "/Users/wangyu/Downloads/datasets/mult_train";
////        String dataAddress = "/Users/wangyu/Downloads/datasets/na_test";
//        String dataAddress = "/Users/wangyu/Downloads/datasets/na_train";
//        Instances data = ClassifierTools.loadData(dataAddress);  // training data
//        System.out.println(data.toSummaryString());
//        ValidateData v = new ValidateData();
//        v.validation(data);
//    }

}
