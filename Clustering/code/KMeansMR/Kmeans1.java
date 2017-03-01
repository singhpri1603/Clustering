import java.io.IOException;
import java.util.*;
import java.lang.Math;
import java.lang.System;
import java.io.PrintWriter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Kmeans1 {

  public static class CentroidMapper
       extends Mapper<Object, Text, IntWritable, Text>{


    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      String[] line = value.toString().split(",");

      Double min = Double.MAX_VALUE;
      int center = 0;
      int c = 0;
      for(Double[] f:centroids){
        Double dist = 0.0;
        for(int i = 0; i<f.length; i++){
          Double temp = Double.parseDouble(line[i]) - f[i];
          temp = Math.pow(temp, 2);
          dist += temp;
        }
        dist = Math.sqrt(dist);


        if(dist < min){
          min = dist;
          center = c;

        }
        c++;
      }
      String result = Arrays.toString(line);
      result = result.replaceAll("\\]","").replaceAll("\\[", "").replaceAll(",","");

      output += center+"\n";
      context.write(new IntWritable(center), new Text(result));
    }
  }

  public static class CentroidReducer
       extends Reducer<IntWritable,Text,DoubleWritable,DoubleWritable> {
    private DoubleWritable result = new DoubleWritable();

    public void reduce(IntWritable key, Iterable<Text> values,
                       Context context
                       ) throws IOException, InterruptedException {
      Double[] avg = new Double[centroids.get(0).length];
      Arrays.fill(avg, 0.0);
      int counter = 0;

      for (Text val : values) {
        String[] line = val.toString().split(" +");
        
        for(int i=0; i<line.length; i++){
          avg[i] += Double.parseDouble(line[i]);
        }
        counter++;
      }

      for(int i = 0; i<avg.length;i++){
        avg[i] = avg[i]/counter;
      }
      if(!Arrays.equals(avg, centroids.get(key.get()))){
        converge = true;
      }
      centroids.set(key.get(), avg);

      //System.out.println(key.get() + ": "+ avg);
      //context.write(key, result);
    }
  }

  public static class RandomMapper
       extends Mapper<Object, Text, DoubleWritable, DoubleWritable>{

    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      randCounter++;      
      if(centroids.size() < k && randCounter%seed == 0){
        String[] line = value.toString().split(",");
        Double[] temp = new Double[line.length];
        for(int i=0; i<line.length; i++){
          temp[i] = Double.parseDouble(line[i]);
        }
        //System.out.println(Arrays.toString(temp));
        centroids.add(temp);
      }
    }
  }

  static List<Double[]> centroids;
  static int randCounter = 0;
  static int seed = 50;  // seed constraint: 1 <= seed <= n/k
  static boolean converge = true;
  static int k;
  static String output = "";
  public static void main(String[] args) throws Exception {

    seed = Integer.parseInt(args[3]);
    long start = System.currentTimeMillis();
    k = Integer.parseInt(args[2]);

    centroids = new ArrayList<Double[]>();
   // centroids.add(new Double[]{1.0 ,2.0, 3.0});
    //centroids.add(new Double[]{7.0, 7.0, 7.0});
    //centroids.add(new Double[]{13.0, 12.0, 14.0});
    Configuration conf = new Configuration();

    int i = 0;
    Job job2 = Job.getInstance(conf, "Random centroids");
    job2.setJarByClass(Kmeans1.class);
    job2.setMapperClass(RandomMapper.class);
    FileInputFormat.addInputPath(job2, new Path(args[0]));
    FileOutputFormat.setOutputPath(job2, new Path(args[1]+i));
    job2.waitForCompletion(true);

    
    while(converge && i < 50){
      output = "";
      converge = false;
      i++;
      Job job = Job.getInstance(conf, "word count");
      job.setJarByClass(Kmeans1.class);
      job.setMapperClass(CentroidMapper.class);
      job.setReducerClass(CentroidReducer.class);
      job.setOutputKeyClass(IntWritable.class);
      job.setOutputValueClass(Text.class);
      FileInputFormat.addInputPath(job, new Path(args[0]));
      FileOutputFormat.setOutputPath(job, new Path(args[1]+i));
      job.waitForCompletion(true);
    }


    try{
      PrintWriter writer = new PrintWriter("visualize/clusters.txt", "UTF-8");
      writer.println(output);
      
      writer.close();
    } catch (Exception e) {
    }
    System.out.println(centroids.size());
    long end = System.currentTimeMillis();
    System.out.println("Time taken = " + (float)(end-start)/1000 + "s");
    System.out.println("Number of iterations: "+ i);

  }

}