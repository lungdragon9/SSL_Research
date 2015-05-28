package SSL_Research;

import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.clusterers.DBScan;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.EuclideanDistance;;


//Upon looking at distances I found a few differnt kind of distance functions that looked interesting
//Euclidean- Normal Distance
//Chebyshev- Chessboard distance
//Minkowski- A combo of Euclidean and Manhatten
//Note though I will only be working with pure numerical data though
public class SAL_C2_SSL 
{
	private static double NumSDAL = 1;
	private static double NUMSDSSL = 1;
	public static boolean DBSCAN_Check = true;
	public static boolean SSL_Check = true;
	private static int numAL = 10;
	private static int MAXSSL = 10;
	public static long seed =  0;
	private static int TotalNumberALAdded =0;
	private static int TotalNumberSSLAdded =0;
	
	public void Runner(String DataSetLoc, String OutputLoc) throws Exception
	{		
		boolean ALDone = false;
		boolean SSLDone = false;
		boolean keepGoing = true;
		
		Instances FullSet = (new DataSource(DataSetLoc)).getDataSet();
		Instances FullSetCopy = (new DataSource(DataSetLoc)).getDataSet();
		Random random = new Random(seed);
		FullSet.randomize(random);
		
		Vote combinedclassifier = new Vote();
		
		int cutoff = (int)((FullSet.numInstances()-10) * .8);
		//10 for Training
		Instances TrainingSet = new Instances(FullSet,0,10);
		Instances TrainingSet_2 = new Instances(FullSet,0,10);
		//80% -10 for Unlabeled
		Instances Unlab = new Instances(FullSet,10,cutoff);
		Instances UnlabNoClass = new Instances(FullSet,10,cutoff);
		//Remaining to Testing
		Instances Testing = new Instances(FullSet,cutoff,(FullSet.numInstances()-10)-cutoff);
		
		double[] TrainingSetPlaces = null;
		if(true)
		{
			FullSetCopy.deleteAttributeAt(FullSetCopy.attribute("class").index());
			TrainingSetPlaces = TrainingSetFindiner(FullSetCopy,10);
		}
		
		FullSet.setClass(FullSet.attribute("class"));
		
		TrainingSet.delete();
		TrainingSet_2.delete();
		
		for(int i =0; i < TrainingSetPlaces.length; i ++)
		{
			System.out.println(FullSet.instance((int)TrainingSetPlaces[i]).classValue());
			TrainingSet.add(FullSet.instance((int)TrainingSetPlaces[i]));
			TrainingSet_2.add(FullSet.instance((int)TrainingSetPlaces[i]));
		}
		
		File file_output = new File(OutputLoc);
        
        file_output.createNewFile();
        
        FileWriter output_write = new FileWriter(file_output.getAbsoluteFile());
        
		TrainingSet.setClass(TrainingSet.attribute("class"));
		TrainingSet_2.setClass(TrainingSet_2.attribute("class"));
		//System.out.println(TrainingSet.instance(0));
		Unlab.setClass(Unlab.attribute("class"));
		Testing.setClass(Testing.attribute("class"));
		
		J48[] classifiers = new J48[2];
		classifiers[0] = new J48();
		classifiers[1] = new J48();
		
		
		classifiers[0].buildClassifier(TrainingSet);
		classifiers[1].buildClassifier(TrainingSet_2);
		
		Evaluation eval_1 = new Evaluation(Testing);
		Evaluation eval_2 = new Evaluation(Testing);
		Evaluation eval_3 = new Evaluation(Testing);
		
		eval_1.evaluateModel(classifiers[0], Testing);
		
		output_write.write("AL ONE \n");
		output_write.write("???????????????????????????????????????????????????????????????????????????????????????????????????????? \n");
		output_write.write("Num Training Instances : " + TrainingSet.numInstances()+"\n");
		output_write.write("Correct : " + eval_1.correct()+"\n");
		output_write.write("Incorrect : " + eval_1.incorrect()+"\n");
		output_write.write("Error rate : " + eval_1.errorRate()+"\n");
		output_write.write("Pct Correct : " +eval_1.pctCorrect()+"\n");
		output_write.write("AUC : " + eval_1.areaUnderROC(1)+"\n");
		output_write.write("done : "+"\n");
		
		eval_2.evaluateModel(classifiers[1], Testing);
		
		output_write.write("SSL ONE \n");
		output_write.write("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n");
		output_write.write("Num Training Instances : " + TrainingSet_2.numInstances()+"\n");
		output_write.write("Correct : " + eval_2.correct()+"\n");
		output_write.write("Incorrect : " + eval_2.incorrect()+"\n");
		output_write.write("Error rate : " + eval_2.errorRate()+"\n");
		output_write.write("Pct Correct : " +eval_2.pctCorrect()+"\n");
		output_write.write("AUC : " + eval_2.areaUnderROC(1)+"\n");
		output_write.write("done : "+"\n");
		
		combinedclassifier.setClassifiers(classifiers);
		
		eval_3.evaluateModel(combinedclassifier, Testing);
		output_write.write("######################################################################################################### \n");
		output_write.write("Combined ONE \n");
		output_write.write("Correct : " + eval_3.correct()+"\n");
		output_write.write("Incorrect : " + eval_3.incorrect()+"\n");
		output_write.write("Error rate : " + eval_3.errorRate()+"\n");
		output_write.write("Pct Correct : " +eval_3.pctCorrect()+"\n");
		output_write.write("AUC : " + eval_3.areaUnderROC(1)+"\n");
		output_write.write("done : "+"\n");
		
		
		while(Unlab.numInstances() > 0 && keepGoing)
		{
			double[][] ConfidenceVector = findConfidenceVector(Unlab,classifiers[0],TrainingSet);	
			
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
						double value = classifiers[1].classifyInstance(Unlab.instance((int)SSL[i]));
	
						adder.setClassValue(adder.classAttribute().value((int)value));
						
						TrainingSet_2.add(adder);
					}
				}
				else
				{
					int coutner =0;
					
					for(int i =SSL.length-1; i >SSL.length-1-MAXSSL; i --)
					{
						Instance adder = Unlab.instance((int)SSL[i]);
						double value = classifiers[1].classifyInstance(Unlab.instance((int)SSL[i]));
	
						adder.setClassValue(adder.classAttribute().value((int)value));
						
						TrainingSet_2.add(adder);
					}
				}
				System.out.println("Added SSL");
				SSLDone = true;
			}
			
			if(AL.length != 0)
			{				
				double numAdd = numAL;
				if(AL.length < numAL)
				{
					numAdd = AL.length;
				}
				NewAL = new double[(int) numAdd];
				Arrays.fill(NewAL, -1);
				
				if(DBSCAN_Check)
				{					
					Instances dummy = getALData(AL,UnlabNoClass);
					DBScan cluster = new DBScan();
					cluster.buildClusterer(dummy);	
					double[] clusters = cluster.returnClusterList();
					int numClusters = cluster.numberOfClusters();
					int numInClusters = countNumInCluster(clusters);
					
					if(numInClusters < numAdd)
					{
						numAdd = numInClusters;
						NewAL = new double[(int) numAdd];
						Arrays.fill(NewAL, -1);
					}
					
					
					if(numClusters != 0)
					{
						int numPerCluster = (int)numAdd/numClusters;
						NewAL = new double[numPerCluster * numClusters];
						Arrays.fill(NewAL, -1);
						
						int coutner =0; 
	
							for(int i=0 ; i < numClusters; i ++)
							{
								for(int k=0; k < numPerCluster; k ++){
									int randLoc = random.nextInt(dummy.numInstances());
									
									if(clusters[randLoc] == i)
									{
										if(SameCheck(NewAL,AL[randLoc]))
										{
											NewAL[coutner++] = AL[randLoc];
										}
										else
										{
											k--;
										}
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
								//double value = classifiers[0].classifyInstance(Unlab.instance((int)NewAL[i]));

								//adder.setClassValue(adder.classAttribute().value((int)value));
								
								TrainingSet.add(adder);
							}
							System.out.println("Added AL");
							ALDone = true;
					}
					else
					{
						System.out.println("No Clusters Found- Reverting to Basic AL");
						
						numAdd = numAL;
						if(AL.length < numAL)
						{
							numAdd = AL.length;
						}
						NewAL = new double[(int) numAdd];
						Arrays.fill(NewAL, -1);
						
						for(int i= 0; i < numAdd; i ++)
						{
							int randLoc = random.nextInt(AL.length);
							if(SameCheck(NewAL,AL[randLoc]))
							{
								NewAL[i] = AL[randLoc];
							}
							else
							{
								i--;
							}
						}
						
						Arrays.sort(NewAL);
						for(int i =NewAL.length-1; i >=0; i --)
						{
							Instance adder = Unlab.instance((int)NewAL[i]);
							//double value = classifiers[0].classifyInstance(Unlab.instance((int)NewAL[i]));

							//adder.setClassValue(adder.classAttribute().value((int)value));
							
							TrainingSet.add(adder);
						}
						System.out.println("Added AL");
						ALDone = true;
					
					}
					
				}
				else
				{
					for(int i= 0; i < numAdd; i ++)
					{
						int randLoc = random.nextInt(AL.length);
						if(SameCheck(NewAL,AL[randLoc]))
						{
							NewAL[i] = AL[randLoc];
						}
						else
						{
							i--;
						}
					}
					
					Arrays.sort(NewAL);
					for(int i =NewAL.length-1; i >=0; i --)
					{
						Instance adder = Unlab.instance((int)NewAL[i]);
						//double value = classifiers[0].classifyInstance(Unlab.instance((int)NewAL[i]));

						//adder.setClassValue(adder.classAttribute().value((int)value));
						
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
			if(ALDone && !SSL_Check)
			{
				int counter =0;
				for(int i=0 ; i < NewAL.length; i++)
				{
					UnlabRemove[counter++] = (int) NewAL[i]; 
				}
			}
			
			Arrays.sort(UnlabRemove);
			
			for(int i = UnlabRemove.length-1; i >= 0; i--)
			{
				try
				{
				Unlab.delete(UnlabRemove[i]);
				UnlabNoClass.delete(UnlabRemove[i]);
				}
				catch(Exception e)
				{
					keepGoing = false;
					break;			
				}
			}
			
			classifiers[0].buildClassifier(TrainingSet);
			classifiers[1].buildClassifier(TrainingSet_2);
			
			
			eval_1 = new Evaluation(Testing);			
			eval_2 = new Evaluation(Testing);
			eval_3 = new Evaluation(Testing);
			
			eval_1.evaluateModel(classifiers[0], Testing);
			
			output_write.write("AL ONE \n");
			output_write.write("???????????????????????????????????????????????????????????????????????????????????????????????????????? \n");
			output_write.write("Num Training Instances : " + TrainingSet.numInstances()+"\n");
			output_write.write("Correct : " + eval_1.correct()+"\n");
			output_write.write("Incorrect : " + eval_1.incorrect()+"\n");
			output_write.write("Error rate : " + eval_1.errorRate()+"\n");
			output_write.write("Pct Correct : " +eval_1.pctCorrect()+"\n");
			output_write.write("AUC : " + eval_1.areaUnderROC(1)+"\n");
			output_write.write("done : "+"\n");
			
			eval_2.evaluateModel(classifiers[1], Testing);
			output_write.write("SSL ONE \n");
			output_write.write("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n");
			output_write.write("Num Training Instances : " + TrainingSet_2.numInstances()+"\n");
			output_write.write("Correct : " + eval_2.correct()+"\n");
			output_write.write("Incorrect : " + eval_2.incorrect()+"\n");
			output_write.write("Error rate : " + eval_2.errorRate()+"\n");
			output_write.write("Pct Correct : " +eval_2.pctCorrect()+"\n");
			output_write.write("AUC : " + eval_2.areaUnderROC(1)+"\n");
			output_write.write("done : "+"\n");

			combinedclassifier.setClassifiers(classifiers);
			
			eval_3.evaluateModel(combinedclassifier, Testing);
			output_write.write("######################################################################################################### \n");
			output_write.write("Combined ONE \n");
			output_write.write("Correct : " + eval_3.correct()+"\n");
			output_write.write("Incorrect : " + eval_3.incorrect()+"\n");
			output_write.write("Error rate : " + eval_3.errorRate()+"\n");
			output_write.write("Pct Correct : " +eval_3.pctCorrect()+"\n");
			output_write.write("AUC : " + eval_3.areaUnderROC(1)+"\n");
			output_write.write("done : "+"\n");
			
			System.out.println(Unlab.numInstances());
			if((!ALDone && !SSLDone) || (Unlab.numInstances() <= numAL + MAXSSL))
			{
				keepGoing = false;
			}
			ALDone = false;
			SSLDone = false;
			
			
		}
		
		System.out.println("Finsihed Overall");
		output_write.close();
		
		TotalNumberALAdded =0;
		TotalNumberSSLAdded =0;
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
			double dummy = Vector[i][0] - mean;
			Var+=Math.pow(dummy,2);
		}
			Var/=(Vector.length-1);
			
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
			//System.out.println(Vector[i][0]);
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
		if(counter ==0)
		{
			double mean = findMean(Vector);
			double newNumSDAL = NumSDAL;
			while(counter ==0)
			{
				newNumSDAL/=2;
				NumAdd = (mean + (findStanDiv(Vector) * newNumSDAL));
				
				for(int i =0; i < Vector.length; i ++)
				{
					if(Vector[i][0] < NumAdd)
					{
						counter++;
					}
				}
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
	
	public static boolean SameCheck(double[] NewAL, double value)
	{
		for(int i=0; i < NewAL.length; i ++)
		{
			if(NewAL[i] == value)
			{
				return false;
			}
		}
		return true;
	}

	public static int countNumInCluster(double clusters[])
	{
		int num =0;
		for(int i=0; i < clusters.length; i ++)
		{
			if(clusters[i] != -1)
			{
				num++;
			}
		}
		return num;
	}
	
	public static double[] TrainingSetFindiner(Instances dummy,int numAdd) throws Exception
	{
		Random random = new Random(seed);
		double[] TrainingPlaces = new double[numAdd];
		Instances TrainingSet = new Instances(dummy);
		
		TrainingSet.delete();
		if(true)
		{					
			DBScan cluster = new DBScan();
			cluster.setEpsilon(2.0);
			cluster.buildClusterer(dummy);	
			double[] clusters = cluster.returnClusterList();
			int numClusters = cluster.numberOfClusters();
			int numInClusters = countNumInCluster(clusters);
			
			if(numInClusters < numAdd)
			{
				numAdd = numInClusters;
				TrainingPlaces = new double[(int) numAdd];
				Arrays.fill(TrainingPlaces, -1);
			}
			
			
			if(numClusters != 0)
			{
				int numPerCluster = (int)numAdd/numClusters;
				TrainingPlaces = new double[numPerCluster * numClusters];
				Arrays.fill(TrainingPlaces, -1);
				
				int coutner =0; 

					for(int i=0 ; i < numClusters; i ++)
					{
						for(int k=0; k < numPerCluster; k ++){
							int randLoc = random.nextInt(dummy.numInstances());
							
							if(clusters[randLoc] == i)
							{
								if(SameCheck(TrainingPlaces,(double)randLoc))
								{
									TrainingPlaces[coutner++] = randLoc;
								}
								else
								{
									k--;
								}
							}
							else
							{
								k--;
							}
						}

					}
					/*
					Arrays.sort(TrainingPlaces);
					
					for(int i =TrainingPlaces.length-1; i >=0; i --)
					{
						Instance adder = dummy.instance((int)TrainingPlaces[i]);
						//double value = classifiers[0].classifyInstance(Unlab.instance((int)NewAL[i]));

						//adder.setClassValue(adder.classAttribute().value((int)value));
						
						TrainingSet.add(adder);
					}
					*/
			}
		}
		return TrainingPlaces;
	}
	}

