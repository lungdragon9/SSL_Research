package SSL_Research;

import java.util.Random;

public class Semi_A_Runner {
	
public static void main(String[] args) throws Exception
{
	Semi_Active_Learning SEAL = new Semi_Active_Learning();
	Random seed = new Random(System.currentTimeMillis());
	SAL_C2_SSL SEAL_2 = new SAL_C2_SSL();
	
	
	for(int i =0; i < 10; i ++)
	{
		long abab = seed.nextLong();
		Semi_Active_Learning.seed = abab;
		Semi_Active_Learning.DBSCAN_Check = false;
		Semi_Active_Learning.SSL_Check = false;
		
		SAL_C2_SSL.seed = abab;
		SAL_C2_SSL.DBSCAN_Check = false;
		SAL_C2_SSL.SSL_Check = false;
		
		
		//SEAL_2.Runner("C:\\WorkSpace\\binary\\kr-vs-kp\\kr-vs-kp.arff", "C:\\WorkSpace\\binary\\kr-vs-kp\\Output_C1_" + i + ".arff");
		
		
		SEAL_2.Runner("C:\\WorkSpace\\binary\\kr-vs-kp\\kr-vs-kp.arff", "C:\\WorkSpace\\binary\\kr-vs-kp\\Output_C1_" + i + ".arff");
		
		SAL_C2_SSL.DBSCAN_Check = true;
		SAL_C2_SSL.SSL_Check = false;
		
		SEAL_2.Runner("C:\\WorkSpace\\binary\\kr-vs-kp\\kr-vs-kp.arff", "C:\\WorkSpace\\binary\\kr-vs-kp\\Output_C2_" + i + ".arff");
		
		SAL_C2_SSL.DBSCAN_Check = false;
		SAL_C2_SSL.SSL_Check = true;
		
		SEAL_2.Runner("C:\\WorkSpace\\binary\\kr-vs-kp\\kr-vs-kp.arff", "C:\\WorkSpace\\binary\\kr-vs-kp\\Output_C3_" + i + ".arff");
		
		SAL_C2_SSL.DBSCAN_Check = true;
		SAL_C2_SSL.SSL_Check = true;
		
		SEAL_2.Runner("C:\\WorkSpace\\binary\\kr-vs-kp\\kr-vs-kp.arff", "C:\\WorkSpace\\binary\\kr-vs-kp\\Output_C4_" + i + ".arff");
		
	}
}
}
