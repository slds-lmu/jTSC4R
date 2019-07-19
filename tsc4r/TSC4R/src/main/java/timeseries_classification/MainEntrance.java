package timeseries_classification;

public class MainEntrance {


    /**
     *
     * @param args:
     *          args for train:
     *            args[0]: dataset path
     *            args[1]: path to store trained classifier
     *            args[2]: the name of classifier
     *            args[3]: flag for cross-validation
     *          args for predict:
     *            args[0]: dataset path
     *            args[1]: classifer path
     */
    public static void main(String[] args) throws Exception {
        TrainAndPredict tp = new TrainAndPredict();
//        tp.train(args);
        tp.predict(args);
    }
}
