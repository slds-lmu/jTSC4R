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
//        String request =args[0];
//        String[] params=Arrays.copyOfRange(args,1,args.length);
//
//        HandleChain train = new Train();
//        HandleChain predict = new Predict();
//        HandleChain trainWithHyper = new TrainWithHyper();
//        HandleChain noHandle = new NoHandle();
//        train.setNext(trainWithHyper);
//        trainWithHyper.setNext(predict);
//        predict.setNext(noHandle);
//        train.process(request,params);
        TrainAndPredict tp = new TrainAndPredict();
        tp.train(args);
//        tp.train(args);
    }
}
