import java.io.*;
import java.util.*;

public class KMeansIrisClustering {
    private static class Iris {
        private double[] features;
        private String type;

        public Iris(double[] features, String type) {
            this.features = features;
            this.type = type;
        }

        public double[] getFeatures() {
            return features;
        }

        public String getType() {
            return type;
        }
    }

    private static class KMeans {
        private List<Iris> data;
        private List<List<Iris>> clusters;
        private double[][] centroids;

        public KMeans(List<Iris> data) {
            this.data = data;
        }

        public void execute(int k) {
            initializeCentroids(k);
            boolean changed;
            do {
                clusters = createClusters(k);
                changed = updateCentroids();
            } while (changed);
        }

        private void initializeCentroids(int k) {
            centroids = new double[k][];
            Random rand = new Random();
            for (int i = 0; i < k; i++) {
                centroids[i] = data.get(rand.nextInt(data.size())).getFeatures().clone();
            }
        }

        private List<List<Iris>> createClusters(int k) {
            List<List<Iris>> clusters = new ArrayList<>();
            for (int i = 0; i < k; i++) {
                clusters.add(new ArrayList<>());
            }
            for (Iris iris : data) {
                int clusterIndex = findClosestCentroid(iris.getFeatures());
                clusters.get(clusterIndex).add(iris);
            }
            return clusters;
        }

        private int findClosestCentroid(double[] features) {
            int index = -1;
            double minDistance = Double.MAX_VALUE;
            for (int i = 0; i < centroids.length; i++) {
                double distance = 0;
                for (int j = 0; j < features.length; j++) {
                    distance += Math.pow(features[j] - centroids[i][j], 2);
                }
                if (distance < minDistance) {
                    minDistance = distance;
                    index = i;
                }
            }
            return index;
        }

        private boolean updateCentroids() {
            boolean changed = false;
            for (int i = 0; i < centroids.length; i++) {
                double[] newCentroid = new double[centroids[0].length];
                if (clusters.get(i).size() == 0) continue; // Avoid division by zero
                for (Iris iris : clusters.get(i)) {
                    double[] features = iris.getFeatures();
                    for (int j = 0; j < features.length; j++) {
                        newCentroid[j] += features[j];
                    }
                }
                for (int j = 0; j < newCentroid.length; j++) {
                    newCentroid[j] /= clusters.get(i).size();
                }
                if (!Arrays.equals(centroids[i], newCentroid)) {
                    centroids[i] = newCentroid;
                    changed = true;
                }
            }
            return changed;
        }

        public List<List<Iris>> getClusters() {
            return clusters;
        }

        public double[][] getCentroids() {
            return centroids;
        }

        public boolean isEachClusterHomogeneous() {
            for (List<Iris> cluster : clusters) {
                String type = cluster.get(0).getType();
                for (Iris iris : cluster) {
                    if (!iris.getType().equals(type)) {
                        return false;
                    }
                }
            }
            return true;
        }
    }

    private static double[] calculateFeatureMeans(List<Iris> data) {
        double[] means = new double[data.get(0).getFeatures().length];
        for (Iris iris : data) {
            double[] features = iris.getFeatures();
            for (int i = 0; i < features.length; i++) {
                means[i] += features[i];
            }
        }
        for (int i = 0; i < means.length; i++) {
            means[i] /= data.size();
        }
        return means;
    }

    private static double[] calculateFeatureStdDevs(List<Iris> data, double[] means) {
        double[] stdDevs = new double[means.length];
        for (Iris iris : data) {
            double[] features = iris.getFeatures();
            for (int i = 0; i < features.length; i++) {
                stdDevs[i] += Math.pow(features[i] - means[i], 2);
            }
        }
        for (int i = 0; i < stdDevs.length; i++) {
            stdDevs[i] = Math.sqrt(stdDevs[i] / data.size());
        }
        return stdDevs;
    }

    private static List<Iris> normalizeData(List<Iris> data) {
        double[] means = calculateFeatureMeans(data);
        double[] stdDevs = calculateFeatureStdDevs(data, means);
        List<Iris> normalizedData = new ArrayList<>();
        for (Iris iris : data) {
            double[] normalizedFeatures = new double[iris.getFeatures().length];
            double[] features = iris.getFeatures();
            for (int i = 0; i < features.length; i++) {
                normalizedFeatures[i] = (features[i] - means[i]) / stdDevs[i];
            }
            normalizedData.add(new Iris(normalizedFeatures, iris.getType()));
        }
        return normalizedData;
    }

    public static void main(String[] args) throws IOException {
        List<Iris> irisData = loadIrisData("iris.txt");
        List<Iris> normalizedIrisData = normalizeData(irisData);

        int k = 1;
        while (true) {
            KMeans kMeans = new KMeans(irisData);
            kMeans.execute(k);
            if (kMeans.isEachClusterHomogeneous()) {
                System.out.println("Minimum clusters without normalization: " + k);
                break;
            }
            k++;
        }

        k = 1;
        while (true) {
            KMeans kMeans = new KMeans(normalizedIrisData);
            kMeans.execute(k);
            if (kMeans.isEachClusterHomogeneous()) {
                System.out.println("Minimum clusters with normalization: " + k);
                break;
            }
            k++;
        }

        KMeans finalKMeans = new KMeans(irisData);
        finalKMeans.execute(k);
        System.out.println("Final centroids without normalization:");
        double[][] centroids = finalKMeans.getCentroids();
        for (int i = 0; i < centroids.length; i++) {
            System.out.println("Centroid " + i + ": " + Arrays.toString(centroids[i]));
        }
        // Print each iris and its assigned cluster
        System.out.println("\nIris data and their cluster assignments:");
        List<List<Iris>> clusters = finalKMeans.getClusters();
        for (int i = 0; i < clusters.size(); i++) {
            List<Iris> cluster = clusters.get(i);
            for (Iris iris : cluster) {
                System.out.println("Iris " + Arrays.toString(iris.getFeatures()) + " (" + iris.getType() + ") -> Cluster " + i);
            }
        }

        KMeans finalKMeansNormalized = new KMeans(normalizedIrisData);
        finalKMeansNormalized.execute(k);
        System.out.println("Final centroids with normalization:");
        double[][] centroidsNormalized = finalKMeansNormalized.getCentroids();
        for (int i = 0; i < centroidsNormalized.length; i++) {
            System.out.println("Centroid " + i + ": " + Arrays.toString(centroidsNormalized[i]));
        }
        // Print each iris and its assigned cluster
        System.out.println("\nIris data and their cluster assignments:");
        List<List<Iris>> clusters2 = finalKMeansNormalized.getClusters();
        for (int i = 0; i < clusters2.size(); i++) {
            List<Iris> cluster = clusters2.get(i);
            for (Iris iris : cluster) {
                System.out.println("Iris " + Arrays.toString(iris.getFeatures()) + " (" + iris.getType() + ") -> Cluster " + i);
            }
        }
    }

    private static List<Iris> loadIrisData(String filePath) throws IOException {
        List<Iris> irisData = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(",");
            double[] features = new double[parts.length - 1];
            for (int i = 0; i < parts.length - 1; i++) {
                features[i] = Double.parseDouble(parts[i]);
            }
            irisData.add(new Iris(features, parts[parts.length - 1]));
        }
        reader.close();
        return irisData;
    }
}
