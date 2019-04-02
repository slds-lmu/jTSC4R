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
        String request =args[0];
        String[] params=Arrays.copyOfRange(args,1,args.length);
//        TrainAndPredict trainAndPredict=new TrainAndPredict();
//        if (train_or_predict.equals("train")){
//            trainAndPredict.train(params);
//        }else if (train_or_predict.equals("predict")){
//            trainAndPredict.predict(params);
//        }
//        else{
//            System.out.println("you should enter train or prediction");
//        }
//        switch (train_or_predict){
//            case "train": System.out.println("train");trainAndPredict.train(params);break;
//            case "predict" : System.out.println("predict");trainAndPredict.predict(params);break;
//            case "trainWithHyper":System.out.println("trainWithHyper");trainAndPredict.trainWithHyper(params);break;
//            default:System.out.println("you should enter train or prediction");
//        }

        HandleChain train = new Train();
        HandleChain predict = new Predict();
        HandleChain trainWithHyper = new TrainWithHyper();
        HandleChain noHandle = new NoHandle();
        train.setNext(trainWithHyper);
        trainWithHyper.setNext(predict);
        predict.setNext(noHandle);
        train.process(request,params);

    }
}
