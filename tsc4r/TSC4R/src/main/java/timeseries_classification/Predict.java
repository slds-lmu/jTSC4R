package timeseries_classification;

import scala.None;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.io.*;

public class Predict implements HandleChain {
    private HandleChain nextInChain;
    @Override
    public void setNext(HandleChain c) {
        nextInChain = c;
    }

    @Override
    public Object process(String[] params) throws Exception {
            String moduleAddress =params[0];
            String dataAddress=params[1];
            Instances testData = ClassifierTools.loadData(dataAddress);
            ValidateData v = new ValidateData();
            boolean f = v.validationForPrediction(testData);
            Classifier c=null;
            //read obj from file
            FileInputStream fi = new FileInputStream(new File(moduleAddress));
            ObjectInputStream oi = new ObjectInputStream(fi);
            try {
                c=(Classifier) oi.readObject();
            } catch (ClassNotFoundException e) {
                System.out.println("there is no classifiers on this path");
            }
            oi.close();
            fi.close();
            double[] results=new double[testData.numInstances()];
            for(int i=0;i<testData.numInstances();i++){
                results[i]=c.classifyInstance(testData.instance(i));
            }
            return results;
    }
}
