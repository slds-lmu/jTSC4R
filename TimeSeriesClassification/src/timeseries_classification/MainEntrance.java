package timeseries_classification;
import java.util.Arrays;

public class MainEntrance {


    /**
     *
     * @param args:
     *            arg[0]:name of Algorithms to be applied
     *            arg[1]:train or prediction
     *            arg[2]...:params of train or prediction
     */
    public static void main(String[] args) throws Exception {
        String train_or_predict =args[0];
        String[] params=Arrays.copyOfRange(args,1,args.length);
        TrainAndPredict trainAndPredict=new TrainAndPredict();
        if (train_or_predict.equals("train")){
            trainAndPredict.train(params);
        }else if (train_or_predict.equals("predict")){
            trainAndPredict.predict(params);
        }else{
            System.out.println("you should enter train or prediction");
        }

    }
}
