package timeseries_classification;

public class NoHandle implements HandleChain {
    @Override
    public void setNext(HandleChain nextInChain) {

    }

    @Override
    public Object process(String[] params) throws Exception {
        System.out.println("There is no such algorithm");
        return null;
    }
}
