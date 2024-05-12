public class Iris {
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
