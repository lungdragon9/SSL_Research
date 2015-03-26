package SSL_Research;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import com.sun.xml.internal.bind.v2.runtime.unmarshaller.XsiNilLoader.Array;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.clusterers.DBScan;
import weka.clusterers.forOPTICSAndDBScan.Databases.Database;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.EuclideanDistance;;


//Upon looking at distances I found a few differnt kind of distance functions that looked interesting
//Euclidean- Normal Distance
//Chebyshev- Chessboard distance
//Minkowski- A combo of Euclidean and Manhatten
//Note though I will only be working with pure numerical data though
public class Semi_Active_Learning 
{
	private static double NumSDAL = 1;
	private static double NUMSDSSL = 1;
	private static boolean DBSCAN_Check = true;
	private static boolean SSL_Check = true;
	private static int numAL = 10;
	private static int MAXSSL = 40;
	private static long seed =  System.currentTimeMillis();
	private static int TotalNumberALAdded =0;
	private static int TotalNumberSSLAdded =0;
	public static void main(String args[]) throws Exception
	{		
		boolean ALDone = false;
		boolean SSLDone = false;
		boolean keepGoing = true;
		
		Instances FullSet = (new DataSource("C:\\WorkSpace\\binary\\breast-w\\breast-w.arff")).getDataSet();
		Random random = new Random(seed);
		FullSet.randomize(random);
		
		int cutoff = (int)((FullSet.numInstances()-10) * .8);
		//10 for Training
		Instances TrainingSet = new Instances(FullSet,0,10);
		//80% -10 for Unlabeled
		Instances Unlab = new Instances(FullSet,10,cutoff);
		Instances UnlabNoClass = new Instances(FullSet,10,cutoff);
		//Remaining to Testing
		Instances Testing = new Instances(FullSet,cutoff,(FullSet.numInstances()-10)-cutoff);
		
		
		
		TrainingSet.setClass(TrainingSet.attribute("Class"));
		//System.out.println(TrainingSet.instance(0));
		Unlab.setClass(Unlab.attribute("Class"));
		Testing.setClass(Testing.attribute("Class"));
		J48  classifier = new J48();
		
		classifier.buildClassifier(TrainingSet);
		
		Evaluation eval_1 = new Evaluation(Testing);			
		eval_1.evaluateModel(classifier, Testing);
		
		System.out.println("Num Training Instances " + TrainingSet.numInstances());
		System.out.println("Correct: " + eval_1.correct());
		System.out.println("Incorrect " + eval_1.incorrect());
		System.out.println("Error rate " + eval_1.errorRate());
		System.out.println("Pct Correct " +eval_1.pctCorrect());
		System.out.println("AUC: " + eval_1.areaUnderROC(1));
		System.out.println("done");
		
		while(Unlab.numInstances() > 0 && keepGoing)
		{
			double[][] ConfidenceVector = findConfidenceVector(Unlab,classifier,TrainingSet);	
			
			ConfidenceVector = sort(ConfidenceVector);
			//Needed?
			double SD = findStanDiv(ConfidenceVector);
			double numAddAL = findMean(ConfidenceVector) - (NumSDAL * SD);
			double NumAddSSL = findMean(ConfidenceVector) + (NUMSDSSL * SD);
			
			double[] NewAL = new double[numAL];
			double[] AL = WorstPer(Unlab,ConfidenceVector,numAddAL);
			//System.out.println(AL.numInstances());
			double[] SSL = BestPer(Unlab,ConfidenceVector,NumAddSSL);
			//System.out.println(SSL.numInstances());
			//System.out.println("BOB");
			
			
			if (SSL_Check && SSL.length != 0)
			{
				if(SSL.length < MAXSSL)
				{
					for(int i =SSL.length-1; i >=0; i --)
					{
						Instance adder = Unlab.instance((int)SSL[i]);
						double value = classifier.classifyInstance(Unlab.instance((int)SSL[i]));
	
						adder.setClassValue(adder.classAttribute().value((int)value));
						
						TrainingSet.add(adder);
					}
				}
				else
				{
					int coutner =0;
					
					for(int i =SSL.length-1; i >SSL.length-1-MAXSSL; i --)
					{
						Instance adder = Unlab.instance((int)SSL[i]);
						double value = classifier.classifyInstance(Unlab.instance((int)SSL[i]));
	
						adder.setClassValue(adder.classAttribute().value((int)value));
						
						TrainingSet.add(adder);
					}
				}
				System.out.println("Added SSL");
				SSLDone = true;
			}
			
			if(AL.length != 0)
			{				
				if(DBSCAN_Check)
				{					
					Instances dummy = getALData(AL,UnlabNoClass);
					DBScan cluster = new DBScan();
					cluster.buildClusterer(dummy);	
					double[] clusters = cluster.returnClusterList();
					int numClusters = cluster.numberOfClusters();
					
					if(numClusters != 0)
					{
						int numPerCluster = numAL/numClusters;
						NewAL = new double[numPerCluster * numClusters];
						int coutner =0; 
	
							for(int i=0 ; i < numClusters; i ++)
							{
								for(int k=0; k < numPerCluster; k ++){
									int randLoc = random.nextInt(dummy.numInstances());
									
									if(clusters[randLoc] == i)
									{
										NewAL[coutner++] = AL[randLoc];
									}
									else
									{
										k--;
									}
								}
		
							}
							
							Arrays.sort(NewAL);
							for(int i =NewAL.length-1; i >=0; i --)
							{
								Instance adder = Unlab.instance((int)NewAL[i]);
								double value = classifier.classifyInstance(Unlab.instance((int)NewAL[i]));

								adder.setClassValue(adder.classAttribute().value((int)value));
								
								TrainingSet.add(adder);
							}
							System.out.println("Added AL");
							ALDone = true;
					}
					else
					{
						System.out.println("No Clusters Found");
					}
					
				}
				else
				{
					for(int i= 0; i < numAL; i ++)
					{
						int randLoc = random.nextInt(AL.length);
						NewAL[i] = AL[randLoc];						
					}
					
					Arrays.sort(NewAL);
					for(int i =NewAL.length-1; i >=0; i --)
					{
						Instance adder = Unlab.instance((int)NewAL[i]);
						double value = classifier.classifyInstance(Unlab.instance((int)NewAL[i]));

						adder.setClassValue(adder.classAttribute().value((int)value));
						
						TrainingSet.add(adder);
					}
					System.out.println("Added AL");
					ALDone = true;
				}
			}
			int unlabRemoveCoutner =0;
			
			if(SSLDone)
			{
				if(SSL.length < MAXSSL)
				{
					unlabRemoveCoutner = SSL.length;
					TotalNumberSSLAdded += SSL.length;
				}
				else
				{
					unlabRemoveCoutner = MAXSSL;
					TotalNumberSSLAdded += MAXSSL;
				}
			}
			
			if(ALDone)
			{
				unlabRemoveCoutner += NewAL.length;
				TotalNumberALAdded += NewAL.length;
			}
			
			int[] UnlabRemove = new int[unlabRemoveCoutner];
			
			if(SSL_Check)
			{
				if(SSL.length < MAXSSL)
				{
					int counter =0;
					if(SSLDone)
					{
						for(int i =SSL.length-1; i >0; i --)
						{
							UnlabRemove[counter++] = (int)SSL[i];
						}
					}
					if(ALDone)
					{
						for(int i=0 ; i < NewAL.length; i++)
						{
							UnlabRemove[counter++] = (int) NewAL[i]; 
						}
					}
				}
				else
				{			
					int counter =0;
					if(SSLDone)
					{
						for(int i =SSL.length-1; i >SSL.length-1-MAXSSL; i --)
						{
							UnlabRemove[counter++] = (int)SSL[i];
						}
					}
					if(ALDone)
					{
						for(int i=0 ; i < NewAL.length; i++)
						{
							UnlabRemove[counter++] = (int) NewAL[i]; 
						}
					}
				}
			}
			
			Arrays.sort(UnlabRemove);
			
			for(int i = UnlabRemove.length-1; i > 0; i--)
			{
				Unlab.delete(UnlabRemove[i]);
				UnlabNoClass.delete(UnlabRemove[i]);
			}
			
			classifier.buildClassifier(TrainingSet);
			
			eval_1 = new Evaluation(Testing);			
			eval_1.evaluateModel(classifier, Testing);
			
			System.out.println("Num Training Instances " + TrainingSet.numInstances());
			System.out.println("Correct: " + eval_1.correct());
			System.out.println("Incorrect " + eval_1.incorrect());
			System.out.println("Error rate " + eval_1.errorRate());
			System.out.println("Pct Correct " +eval_1.pctCorrect());
			System.out.println("AUC: " + eval_1.areaUnderROC(1));
			System.out.println("Number AL Added " + TotalNumberALAdded);
			System.out.println("Number SSL Added " + TotalNumberSSLAdded);
			System.out.println("done");
			
			if(!ALDone && !SSLDone)
			{
				keepGoing = false;
			}
			ALDone = false;
			SSLDone = false;
			
			
		}
		
		
	}
	
	
	//Give both Training and Unlabled
	//Note Will need to check to see if that if I need to do a -2 on line dist = (Instances.numInstances()-1/dist);
	public static double findDistance(Instances Instances)
	{
		double dist=0;
		EuclideanDistance distanceFinder = new EuclideanDistance(Instances);
		double [] Weights = new double[Instances.numAttributes()];
		Arrays.fill(Weights,1);
		distanceFinder.setWeights(Weights);
		
		for(int i =0; i < Instances.numInstances()-1; i ++)
		{
			dist += (1/distanceFinder.distance(Instances.instance(i), Instances.instance(Instances.numInstances()-1)));
		}
		dist = (Instances.numInstances()-1/dist);
		
		return dist;
	}
	

	public static double[][] findConfidenceVector(Instances Unlab, J48 classifier, Instances TrainingSet) throws Exception
	{
		double[][] ConfidenceVector = new double[Unlab.numInstances()][2];
		Instances OneSet= new Instances(TrainingSet,0,TrainingSet.numInstances());
		Instances ZeroSet = new Instances(TrainingSet,0,TrainingSet.numInstances());
		double[] weights = new double[Unlab.numInstances()];
		
		for(int i =OneSet.numInstances(); i > 0; i--)
		{
			//System.out.println(TrainingSet.instance(i-1));
			if(OneSet.instance(i-1).classValue() == 1)
			{
				ZeroSet.delete(i-1);
			}
			else
			{
				OneSet.delete(i-1);
			}
		}
		
		for(int i =0; i < Unlab.numInstances(); i++)
		{
			double[] Distribution = classifier.distributionForInstance(Unlab.instance(i));
			//System.out.println(Unlab.instance(i));
			
			if(Distribution[0] > Distribution[1])
			{
				Instances dummy = new Instances(ZeroSet,0,ZeroSet.numInstances());
				dummy.add(Unlab.instance(i));
				double Weight = findDistance(dummy);
				weights[i] = Weight;
				
				ConfidenceVector[i][0] = Distribution[0];
				ConfidenceVector[i][1] = i;
			}
			else
			{
				Instances dummy = new Instances(OneSet,0,OneSet.numInstances());
				dummy.add(Unlab.instance(i));
				double Weight = findDistance(dummy);
				weights[i] = Weight;
				
				ConfidenceVector[i][0] = Distribution[1];
				ConfidenceVector[i][1] = i;
			}
		}
		double min =Double.MAX_VALUE, max = Double.MIN_VALUE;
		
		for(int i =0; i < weights.length; i ++)
		{
			if(min > weights[i])
			{
				min = weights[i];
			}
			if(max < weights[i])
			{
				max = weights[i];
			}
		}
		
		for(int i =0; i < weights.length; i ++)
		{
			weights[i] = 1+((weights[i] - max) / (min-max));
		}
		
		for(int i =0; i < ConfidenceVector.length; i ++)
		{
			ConfidenceVector[i][0] = ConfidenceVector[i][0] * weights[i];
		}
		return ConfidenceVector;
		
	}

	public static double[][] sort(double [][] Vector)
	{
		
		java.util.Arrays.sort(Vector, new java.util.Comparator<double[]>() {
		    public int compare(double[] a, double[] b) {
		        return Double.compare(a[0], b[0]);
		    }
		});
		
		return Vector;
	}
	
	//Is this needed? Since we are just using Stand Div to find the top and bottom 2%?
	public static double findStanDiv(double[][] Vector)
	{
		double mean = findMean(Vector);
		
		double Var = 0;
		
		for(int i =0; i < Vector.length; i ++)
		{
			Var+=Math.pow((mean - Vector[i][0]),2);
		}
			Var/=(Vector.length);
			
		double SD =0;
		
		SD = Math.sqrt(Var);
		
		return SD;
	}
	
	public static double findMean(double[][] Vector)
	{
		double mean =0;
		//System.out.println(Vector.length);
		
		for(int i =0; i < Vector.length; i ++)
		{
			mean +=Vector[i][0];
		}
			mean/=Vector.length;
			
		return mean;
	}

	public static double[] WorstPer(Instances Unlab, double[][] Vector, double NumAdd)
	{
		int counter =0;
		
		for(int i =0; i < Vector.length; i ++)
		{
			if(Vector[i][0] < NumAdd)
			{
				counter++;
			}
		}
		double[] UnlabReturn = new double[counter];
		for(int i =0; i < Vector.length; i ++)
		{
			if(Vector[i][0] < NumAdd)
			{
				UnlabReturn[i] = Vector[i][1];
			}
		}
		return UnlabReturn;
	}
	
	public static double[] BestPer(Instances Unlab, double[][] Vector, double NumAdd)
	{
		int counter =0;
		
		for(int i =0; i < Vector.length; i ++)
		{
			if(Vector[i][0] > NumAdd)
			{
				counter++;
			}
		}
		double[] UnlabReturn = new double[counter];
		counter =0;
		for(int i =0; i < Vector.length; i ++)
		{
			if(Vector[i][0] > NumAdd)
			{
				UnlabReturn[counter++] = Vector[i][1];
			}
		}
		return UnlabReturn;
	}

	public static Instances getALData(double[] AL, Instances Unlab)
	{
		Instances dummy = new Instances(Unlab,0,1);
		dummy.delete();
		
		for(int i =0; i < AL.length; i ++)
		{
			dummy.add(Unlab.instance((int)AL[i]));
		}
		
		return dummy;
	}
}
