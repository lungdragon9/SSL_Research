package SSL_Research;

import java.util.Arrays;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Co_Training {
	public static void main(String args[])
	{
		//Have to put it all ain the Try method to for allot of the WEKA functions
		try {
			Instances FullSet = (new DataSource("C:\\WorkSpace\\binary\\breast-w\\breast-w.arff")).getDataSet();
			Random random = new Random(System.currentTimeMillis());
			
			//Randomize the fullset to allow better testing. Will later set seed to allow even testing among 3 methods
			FullSet.randomize(random);
			
			//Need to randomize FullSet
			int cutoff = (int)((FullSet.numInstances()-10) * .8);
			//10 for Training
			Instances TrainingSet1 = new Instances(FullSet,0,10);
			Instances TrainingSet2 = new Instances(FullSet,0,10);
			//80% -10 for Unlabeled
			Instances Unlab1 = new Instances(FullSet,10,cutoff);
			Instances Unlab2 = new Instances(FullSet,10,cutoff);
			//Remaining to Testing
			Instances Testing1 = new Instances(FullSet,cutoff,(FullSet.numInstances()-10)-cutoff);
			Instances Testing2 = new Instances(FullSet,cutoff,(FullSet.numInstances()-10)-cutoff);
			
			//Must designate the Class attribute
			//Co-Training requires a split among the Attributes so 2 models are built and 2 sets for each set is needed
			TrainingSet1.setClass(TrainingSet1.attribute("Class"));
			TrainingSet2.setClass(TrainingSet2.attribute("Class"));
			Unlab1.setClass(Unlab1.attribute("Class"));
			Unlab2.setClass(Unlab2.attribute("Class"));
			Testing1.setClass(Testing1.attribute("Class"));
			Testing2.setClass(Testing2.attribute("Class"));
			
			int AttributeOrder[];
			AttributeOrder = RandOrder((TrainingSet1.numAttributes()-1),System.currentTimeMillis());
			//Testing
			//System.out.println(TrainingSet1.numAttributes());
			
			TrainingSet1 = SplitFirstSet(AttributeOrder,TrainingSet1);
			Unlab1 = SplitFirstSet(AttributeOrder,Unlab1);
			Testing1 = SplitFirstSet(AttributeOrder,Testing1);
			
			TrainingSet2 = SplitSecondSet(AttributeOrder,TrainingSet2);
			Unlab2 = SplitSecondSet(AttributeOrder,Unlab2);
			Testing2 = SplitSecondSet(AttributeOrder,Testing2);
			
			System.out.println(TrainingSet2.numAttributes());
			System.out.println(TrainingSet1.instance(0));
			System.out.println(TrainingSet2.instance(0));
			
			/*
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
			*/
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	//Gets a random order for the Attributes
	public static int[] RandOrder(int NumAttributes,long Seed)	
	{
		Random rand = new Random(Seed);
		int Order[] = new int[NumAttributes];
		Arrays.fill(Order, -1);
		
		for(int i =0; i < NumAttributes; i++)
		{
			int num = rand.nextInt(NumAttributes);
			if(NotChosenBefore(num,Order))
			{
				Order[i] = num;
			}
			else
			{
				i--;
			}
		}
		
		return Order;
	}
	
	//Makes sure that a number isn't chosen before
	public static boolean NotChosenBefore(int num, int Order[])
	{
		for(int i =0; i < Order.length; i ++)
		{
			if(num == Order[i])
			{
				return false;
			}
		}
		return true;
	}
	
	//Removes Attributes from a set
	public static Instances SplitFirstSet(int Order[], Instances set)
	{
		
		int firstHalf[] = new int[Order.length/2];
		
		for(int i =0; i< Order.length/2; i++)
		{
			firstHalf[i] = Order[i];			
		}
		
		Arrays.sort(firstHalf);
		
		for(int i =firstHalf.length; i > 0; i--)
		{
			set.deleteAttributeAt(firstHalf[i-1]);
		}
		return set;
	}
	//Removes the second half of attributes
	public static Instances SplitSecondSet(int Order[], Instances set)
	{
		int NumForSecondArray =0;
		int numRuns =(Order.length/2);
		int SizeOfDelete = (Order.length/2);
		
		if(Order.length %2 != 0)
		{
			SizeOfDelete++;
		}
		
		int SecondHalf[] = new int[SizeOfDelete];
		
		for(int i =Order.length; i> numRuns; i--)
		{
			SecondHalf[NumForSecondArray++] = Order[i-1];			
		}
		
		Arrays.sort(SecondHalf);
		
		for(int i =(SecondHalf.length); i > 0; i--)
		{
			set.deleteAttributeAt(SecondHalf[i-1]);
		}
		return set;
	}
	
	
}
