package timeseries_classification;


public class TrainAndPredict {
    /**
     *
     * @param args
     *            args[0]: dataset path
     *            args[1]: path to store trained classifier
     *            args[2]: the name of classifier
     *            args[3]: flag for cross-validation
     * @return the trained classifier object
     * @throws Exception
     */
    public Object train(String[] args) throws Exception {

        HandleChain train = new Train();
        HandleChain trainWithHyper = new TrainWithHyper();
        HandleChain noHandle = new NoHandle();
        train.setNext(trainWithHyper);
        trainWithHyper.setNext(noHandle);
        Object o =train.process(args);
        return o;
    }

    /**
     *
     * @param args
     *            args[0]: dataset path
     *            args[1]: classifer path
     * @return classifier object
     * @throws Exception
     */
    public Object predict(String[] args) throws Exception {
        HandleChain predict = new Predict();
        Object o = predict.process(args);
        return o;
    }

}
