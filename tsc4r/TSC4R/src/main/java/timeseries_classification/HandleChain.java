package timeseries_classification;

import java.io.FileNotFoundException;

public interface HandleChain {
    public abstract void setNext(HandleChain nextInChain);
    public abstract Object process(String[] params) throws Exception;
}
