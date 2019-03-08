
package utilities;

import java.io.File;
import java.io.FileFilter;
import java.io.FilenameFilter;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class FileHandlingTools {

    /**
     * List the directories contained in the directory given
     */
    public File[] listDirectories(String baseDirectory) {
        return (new File(baseDirectory)).listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                return pathname.isDirectory();
            }
        });
    }

    /**
     * List the directories contained in the directory given
     */
    public String[] listDirectoryNames(String baseDirectory) {
        return (new File(baseDirectory)).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return dir.isDirectory();
            }
        });
    }
    
     /**
     * List the files contained in the directory given
     */
    public File[] listFiles(String baseDirectory) {
        return (new File(baseDirectory)).listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                return pathname.isFile();
            }
        });
    }
    
     /**
     * List the files contained in the directory given
     */
    public String[] listFileNames(String baseDirectory) {
        return (new File(baseDirectory)).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return dir.isFile();
            }
        });
    }
    
     /**
     * List the files contained in the directory given, that end with the given suffix (file extension, generally)
     */
    public File[] listFilesEndingWith(String baseDirectory, String suffix) {
        return (new File(baseDirectory)).listFiles(new FileFilter() {
            @Override
            public boolean accept(File pathname) {
                return pathname.isFile() && pathname.getName().endsWith(suffix);
            }
        });
    }
    
     /**
     * List the files contained in the directory given, that end with the given suffix (file extension, generally)
     */
    public String[] listFileNamesEndingWith(String baseDirectory, String suffix) {
        return (new File(baseDirectory)).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return dir.isFile() && name.endsWith(suffix);
            }
        });
    }
    
}
