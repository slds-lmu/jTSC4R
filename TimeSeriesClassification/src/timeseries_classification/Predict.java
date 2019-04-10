package timeseries_classification;

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
    public Object process(String request, String[] params) throws Exception {
        if(request.equals("predict")){
            String moduleAddress =params[0];
            String dataAddress=params[1];
            Instances testData = ClassifierTools.loadData(dataAddress);
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

        }else{
            Object o = nextInChain.process(request,params);
            return o;
        }
    }
}
