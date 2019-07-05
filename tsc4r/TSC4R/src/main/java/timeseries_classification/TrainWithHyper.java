package timeseries_classification;

import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Type;

public class TrainWithHyper implements HandleChain{
    private HandleChain nextInChain;
    @Override
    public void setNext(HandleChain c) {
        nextInChain = c;

    }

    @Override
    public Object process(String request, String[] params) throws Exception {
        if(request.equals("train")&&params.length>4){
            String dataAddress=params[0];
            String moduleAddress=params[1];
            String classiferName=params[2];
            String cvFlag = params[3];
            String[] hyperparams = params[4].split("_");

            Instances data = ClassifierTools.loadData(dataAddress);  // training data
            ValidateData v = new ValidateData();
            v.validation(data);
            Object c=null;
            try {
                Class algorithm=Class.forName(classiferName);  // reflection out the class
                c=algorithm.newInstance();
                Method[] methods=algorithm.getMethods();

                for (int i = 0; i<hyperparams.length; ){
                    // get the method name to set specific hyper params
                    String methodName = hyperparams[i];
                    for (Method method : methods){
                        if(method.getName().equals(methodName)){
                            Type[] types=method.getParameterTypes();
                            for (Type type : types){
                                switch (type.getTypeName()){
                                    case "float":{
                                        float value = Float.valueOf(hyperparams[i+1]);
                                        method.invoke(c,value);
                                        break;
                                    }
                                    case "int":{
                                        int value = Integer.valueOf(hyperparams[i+1]);
                                        method.invoke(c,value);
                                        break;

                                    }
                                    case "java.lang.String":{
                                        String value = hyperparams[i+1];
                                        method.invoke(c,value);
                                        break;
                                    }
                                    default:
                                }
                            }
                        }
                    }
                    i++;
                }
                Method builderClassifier=algorithm.getMethod("buildClassifier",data.getClass());
                // buidClassifier is a method shared across TSC
                builderClassifier.invoke(c,data);
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            } catch (InvocationTargetException e) {
                e.printStackTrace();
            } catch (InstantiationException e) {
                e.printStackTrace();
            } catch (NoSuchMethodException e) {
                e.printStackTrace();
            }
            if(cvFlag.equals("1")){
                double[][] a=ClassifierTools.crossValidationWithStats((Classifier) c,data, 10);
                System.out.println("ROTF ACC = "+a[0][0]);
            }

            //write obj to file
            FileOutputStream f =new FileOutputStream(new File(moduleAddress));
            ObjectOutputStream o =new ObjectOutputStream(f);
            o.writeObject(c);
            o.close();
            f.close();
            return c;

        }else {
            Object o = nextInChain.process(request,params);
            return o;
        }
    }
}
