package timeseries_classification;
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

}
