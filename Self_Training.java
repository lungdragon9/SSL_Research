package SSL_Research;

import java.util.Random;

import sun.security.jca.GetInstance.Instance;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Self_Training {

	public static void main(String[] args)
	{
		//Have to put it all ain the Try method to for alot of the WEKA functions
		try {
			Instances FullSet = (new DataSource("C:\\WorkSpace\\binary\\breast-w\\breast-w.arff")).getDataSet();
			Random random = new Random(System.currentTimeMillis());
			
			//Randomize the fullset to allow better testing. Will later set seed to allow even testing among 3 methods
			FullSet.randomize(random);
			
			//Need to randomize FullSet
			int cutoff = (int)((FullSet.numInstances()-10) * .8);
			//10 for Training
			Instances TrainingSet = new Instances(FullSet,0,10);
			//80% -10 for Unlabeled
			Instances Unlab = new Instances(FullSet,10,cutoff);
			//Remaining to Testing
			Instances Testing = new Instances(FullSet,cutoff,(FullSet.numInstances()-10)-cutoff);
			
			//Must designate the Class attribute
			TrainingSet.setClass(TrainingSet.attribute("Class"));
			Unlab.setClass(Unlab.attribute("Class"));
			Testing.setClass(Testing.attribute("Class"));
			
			//Tree Classifier
			J48 Classifier = new J48();			
			Classifier.buildClassifier(TrainingSet);			
			
			//Used to evaluate the Classifier
			Evaluation eval_1 = new Evaluation(Testing);			
			eval_1.evaluateModel(Classifier, Testing);
			
			System.out.println("Num Training Instances " + TrainingSet.numInstances());
			System.out.println("Correct: " + eval_1.correct());
			System.out.println("Incorrect " + eval_1.incorrect());
			System.out.println("Error rate " + eval_1.errorRate());
			System.out.println("Pct Correct " +eval_1.pctCorrect());
			System.out.println("done");
			
			int numInstances = Unlab.numInstances();
			for(int i =0; i < numInstances; i ++)
			{
				int location = FindBestInstance(Classifier,Unlab);
				//System.out.println(location);
				//System.out.println(Unlab.numInstances());
				
				//Finds Confidence of best prediction
				double Confidence[];
				Confidence = Classifier.distributionForInstance(Unlab.instance(0));
				weka.core.Instance add = Unlab.instance(location);
				
				//Add Instance to training set
				if(Confidence[0] > Confidence[1])
				{
					add.setClassValue("benign");
					System.out.println("1");
					TrainingSet.add(add);
				}
				else
				{
					add.setClassValue("malignant");
					System.out.println("2");
					TrainingSet.add(add);
				}
				
				Unlab.delete(location);
				
				//Rebuilds Trainingset
				Classifier.buildClassifier(TrainingSet);
			}
			
			Classifier.classifyInstance(Testing.instance(0));
			
			
			Evaluation eval_3 = new Evaluation(Testing);
			eval_3.evaluateModel(Classifier, Testing);
			
			System.out.println("Num Training Instances " + TrainingSet.numInstances());
			System.out.println("Correct: " + eval_3.correct());
			System.out.println("Incorrect " + eval_3.incorrect());
			System.out.println("Error rate " + eval_3.errorRate());
			System.out.println("Pct Correct " +eval_3.pctCorrect());
			System.out.println("done");
			
			
		
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private static int FindBestInstance(J48 Classifier, Instances Unlab) throws Exception
	{
		int location = 0;
		double Confidence[];
		Confidence = Classifier.distributionForInstance(Unlab.instance(0));
		double bestConfidence = Best(Confidence);
		double help;
		
		for(int i =1; i < Unlab.numInstances(); i ++)
		{
			Confidence = Classifier.distributionForInstance(Unlab.instance(0));
			help = Best(Confidence);
			
			if(help > bestConfidence)
			{
				bestConfidence = help;
				location =i;
			}
		}
		
		return location;
	}
	public static double Best(double[] Confidences)
	{
		if(Confidences[0] > Confidences[1])
		{
			return Confidences[0];
		}
		else
		{
			return Confidences[1];
		}
		
	}
}
