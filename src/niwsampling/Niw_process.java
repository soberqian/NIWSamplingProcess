package niwsampling;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
/***EM algorithm 
 * @author Qianyang
 * ****/
public class Niw_process {
	public static void main(String args[]){
		HashMap<String, ArrayList<String>> result = new HashMap<String, ArrayList<String>>();
		ArrayList<String> list = new ArrayList<String>();
		ArrayList<String> list1 = new ArrayList<String>();
		list.add("1800070251	13	5	-0.00588 0.0565 0.420 0.681 0.511 0.342 -0.122 0.167 0.386 0.618 ");
		list.add("1800070251	12	5	-0.869 -0.602 0.138 0.749 -0.00474 -0.159 0.358 1.06 0.345 0.399");
		list.add("1808406133	13	5	-0.104 -0.0856 0.542 -0.0631 0.342 -0.273 0.147 -0.196 -0.231 -0.133 ");
		list.add("1800070251	12	5	-0.228 0.171 -0.265 0.490 -0.0555 0.185 -0.517 0.463 0.443 0.786");
		list.add("1808406133	9	4	0.357 0.0482 0.126 -0.177 0.0512 -0.0431 -0.286 0.279 0.243 0.0443 ");
		list.add("1808406133	5	3	0.104 0.644 0.242 0.117 0.274 -0.284 -0.466 0.816 0.0122 0.217 ");
		list.add("1800340094	9	7	0.357 0.0482 0.126 -0.177 0.0512 -0.0431 -0.286 0.279 0.243 0.0443 ");
		list.add("1800340094	5	1	0.104 0.644 0.242 0.117 0.274 -0.284 -0.466 0.816 0.0122 0.217 ");
		result.put("1800340094", list);
		list1.add("1800070251	13	5	-0.00588 0.0565 0.420 0.681 0.511 0.342 -0.122 0.167 0.386 0.618 ");
		list1.add("1800070251	12	5	-0.869 -0.602 0.138 0.749 -0.00474 -0.159 0.358 1.06 0.345 0.399");
		list1.add("1808406133	13	5	-0.104 -0.0856 0.542 -0.0631 0.342 -0.273 0.147 -0.196 -0.231 -0.133 ");
		list1.add("1800070251	12	5	-0.228 0.171 -0.265 0.490 -0.0555 0.185 -0.517 0.463 0.443 0.786");
		list1.add("1808406133	9	4	0.357 0.0482 0.126 -0.177 0.0512 -0.0431 -0.286 0.279 0.243 0.0443 ");
		list1.add("1808406133	5	3	0.104 0.644 0.242 0.117 0.274 -0.284 -0.466 0.816 0.0122 0.217 ");
		list1.add("1800340094	9	7	0.357 0.0482 0.126 -0.177 0.0512 -0.0431 -0.286 0.279 0.243 0.0443 ");
		list1.add("1800340094	5	1	0.104 0.644 0.242 0.117 0.274 -0.284 -0.466 0.816 0.0122 0.217 ");
		result.put("1800070251", list1);
		Map<String, RealMatrix> EMGetW = EMGetW(result);
		for (HashMap.Entry<String, RealMatrix>  entry : EMGetW.entrySet()) {
			
			System.out.println(entry.getKey() + "==" + entry.getValue());
		}
//		System.out.println(EMGetW);
	}
	
	public static Map<String, RealMatrix> EMGetW(HashMap<String, ArrayList<String>> result){
		
		Hashtable<String, Hashtable<Integer,RealMatrix>> xproduct_product_userfeature = new Hashtable<String, Hashtable<Integer,RealMatrix>>();
		Hashtable<String, Hashtable<Integer,RealMatrix>> y5score = new Hashtable<String, Hashtable<Integer,RealMatrix>>();
		for (HashMap.Entry<String, ArrayList<String>> entry : result.entrySet()) {
			//X向量
			Hashtable<Integer,List<double[]>> product_userfeature = new Hashtable<Integer,List<double[]>>();
			//y向量
			Hashtable<Integer,List<Double>> y5score_productlist = new Hashtable<Integer,List<Double>>();
			for (int i = 0; i < entry.getValue().size(); i++) {
				//x向量
				if (product_userfeature.containsKey(Integer.parseInt(entry.getValue().get(i).split("\t")[0]))) {
					List<double[]> userfeatuer = product_userfeature.get(Integer.parseInt(entry.getValue().get(i).split("\t")[0]));
					userfeatuer.add(StringtoDouble(entry.getValue().get(i).split("\t")[3].split(" ")));
					product_userfeature.put(Integer.parseInt(entry.getValue().get(i).split("\t")[0]),userfeatuer);
					//y
					List<Double> y5socre = y5score_productlist.get(Integer.parseInt(entry.getValue().get(i).split("\t")[0]));
					y5socre.add(Double.valueOf(entry.getValue().get(i).split("\t")[2]));
					y5score_productlist.put(Integer.parseInt(entry.getValue().get(i).split("\t")[0]), y5socre);
				}else{
					List<double[]> userfeatuer = new ArrayList<double[]>();
					userfeatuer.add(StringtoDouble(entry.getValue().get(i).split("\t")[3].split(" ")));
					product_userfeature.put(Integer.parseInt(entry.getValue().get(i).split("\t")[0]), userfeatuer);
					//y向量
					List<Double> y5socre = new ArrayList<Double>();
					y5socre.add(Double.valueOf(Double.valueOf(entry.getValue().get(i).split("\t")[2])));
					y5score_productlist.put(Integer.parseInt(entry.getValue().get(i).split("\t")[0]), y5socre);
				}
				
			}
			Hashtable<Integer,RealMatrix> maptoarr = StringarrtoDouble(product_userfeature);
			xproduct_product_userfeature.put(entry.getKey(), maptoarr);
			Hashtable<Integer,RealMatrix> ymaptoarr = y_arrtoDouble(y5score_productlist);
			y5score.put(entry.getKey(), ymaptoarr);
		}
		//定义自由度
		double df = 10;
		double b [][] = new double[10][10];
		for(int i = 0; i < b.length; i++) {
			b[i][i] = 1;  
		}
		//定义尺度矩阵
		RealMatrix x = new Array2DRowRealMatrix(b);
		Niw_process cc = new Niw_process();
		//sample from inverse wishart distribution 从  inverse wishart中采样
		RealMatrix cccc = cc.sampleFromInverseWishartDistribution(df, x);
		double [] c = new double[10];
		for (int i = 0; i < c.length; i++) {
			c[i] = 0;
		}
		double [] dd = sampleFromMultivariateDistribution(c,cccc.getData());
		//mean初始值由多元正太分布采样
		double [] meaninitial = sampleFromMultivariateDistribution(c,cccc.getData());
		//covariance input初始值
		RealMatrix  spmpling_inverse_wishartinitial = cc.sampleFromInverseWishartDistribution(df, x);
		//input xmatrix
		double varianceinitial = 0.2;
		double epsilon= 0.001;
		//EM算法
		Map<String, Map<Integer, RealMatrix>> one_product_xmatrix = new HashMap<String, Map<Integer, RealMatrix>> ();  
		for( String str : xproduct_product_userfeature.keySet() ){
			Map<Integer,RealMatrix > xMatrixList=xproduct_product_userfeature.get(str);
			System.out.println("====================="+xMatrixList.size());
			Map<Integer,RealMatrix > ymatrix = y5score.get(str);
			Map<Integer, RealMatrix> product_xmatrix = expectationMaximizationUpdating(meaninitial,spmpling_inverse_wishartinitial,varianceinitial,xMatrixList,ymatrix,epsilon);
			one_product_xmatrix.put(str, product_xmatrix);
		}
		Map<String, RealMatrix> keyproduct = new HashMap<String, RealMatrix>();
		for (String str : one_product_xmatrix.keySet()) {
			RealMatrix EMGetW = one_product_xmatrix.get(str).get(Integer.parseInt(str));
			keyproduct.put(str, EMGetW);
		}
		return keyproduct;
		
	}

	/**EMupdating 
	 * EM算法迭代
	 * 
	 * */
	static int iter=0;
	public static Map<Integer, RealMatrix> expectationMaximizationUpdating(double []  meaninitial,RealMatrix spmpling_inverse_wishartinitial,double varianceinitial,Map<Integer,RealMatrix > xMatrixList,Map<Integer,RealMatrix > ymatrix,double epsilon) {
		//总体均值 mean input
		double [] mean=meaninitial;
		//总体协方差 covariance input
		RealMatrix   spmpling_inverse_wishart=spmpling_inverse_wishartinitial;
		//EStep
		Map<Integer,List<RealMatrix>> edata=CalculateExpectaction(mean,spmpling_inverse_wishart,varianceinitial,xMatrixList,ymatrix);
		Map<Integer,RealMatrix > wMatrixList=new HashMap<Integer,RealMatrix >();
		Map<Integer,RealMatrix > covariancematrixlist=new HashMap<Integer,RealMatrix >();
		for( int itemnumber : edata.keySet() ){
			wMatrixList.put(itemnumber, edata.get(itemnumber).get(0));
			covariancematrixlist.put(itemnumber, edata.get(itemnumber).get(1));
		}
		//Mstep
		MStepData mStepData=CalculateMaximization(wMatrixList,covariancematrixlist,xMatrixList,ymatrix);
		RealMatrix meanupdate=mStepData.getMeanupdate();
		RealMatrix covarianceupdate=mStepData.getCovarianceupdate();
		double varianceupdate=mStepData.getVarianceupdate();
		System.out.println(mStepData.getVarianceupdate());
		//获取单个产品的w矩阵及方差矩阵
		MStepData wMatrixandvariance=new MStepData();
		Map<Integer,MStepData > wMatrixandvariancemap=new HashMap<Integer,MStepData >();
		if (Math.abs(varianceupdate - varianceinitial) < epsilon) {
			System.out.println("meanupdate:"+meanupdate+"\tcovarianceupdate:"+covarianceupdate+"\tvarianceupdate:"+varianceupdate);
			for( int itemnumber : edata.keySet() ){
				System.out.println(itemnumber+":\t w:"+wMatrixList.get(itemnumber));
			}
			
		}else{
			iter++;
			System.out.println("the current iter:\t"+iter);
			meaninitial=meanupdate.getColumnVector(0).toArray();
			spmpling_inverse_wishartinitial=covarianceupdate;
			varianceinitial=varianceupdate;
			expectationMaximizationUpdating(meanupdate.getColumnVector(0).toArray(),covarianceupdate,varianceupdate,xMatrixList,ymatrix,epsilon);
		}
		
		return wMatrixList;
	}
	/*
	 * 
	 * sample from inverse wishart distribution
	 * 
	 * */ 
	public RealMatrix sampleFromInverseWishartDistribution(double df, RealMatrix scaleMatrix){
		RealMatrix inverseMatrix = inverseMatrix(scaleMatrix);   // 计算尺度矩阵的逆矩阵
		//double a [][] = inverseMatrix.getData();
		//a = IOUtils.convertMatrixToSymmetry(a);
		RealMatrix symmetricInverseMatrix = convertMatrixToSymmetry(inverseMatrix);
		//RealMatrix new_inverseMatrix = new Array2DRowRealMatrix(a);
		for (int i = 0; i < 1000; i++) {
			try {
				RealMatrix samples = sampleFromWishartDistribution(df, symmetricInverseMatrix);
				// 求逆矩阵
				RealMatrix inverse_samples = inverseMatrix(samples);
				RealMatrix symmetric_inverse_samples = convertMatrixToSymmetry(inverse_samples);
				return symmetric_inverse_samples;
			} catch (SingularMatrixException ex) {
				ex.printStackTrace();
			}
		}
		throw new RuntimeException("Unable to generate inverse wishart samples!");
	}

	/*
	 * 
	 * inverse of a matrix
	 * 
	 * */ 

	public static RealMatrix inverseMatrix(RealMatrix A) {
		RealMatrix result = new LUDecomposition(A).getSolver().getInverse();
		return result; 
	}

	public static RealMatrix convertMatrixToSymmetry (RealMatrix c) {
		double [][] a = c.getData();
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < i; j++) {
				if (Math.abs(a[i][j] - a[j][i]) < 0.0001) {
					a[j][i] = a[i][j];
				}
			}
		}
		RealMatrix b = new Array2DRowRealMatrix(a);
		return b;
	}
	public static RealMatrix sampleFromWishartDistribution(double df, RealMatrix scaleMatrix){
		RandomGenerator random = new Well19937c();
		int dim = scaleMatrix.getColumnDimension(); // get the dimension of the matrix
		GammaDistribution [] gammas = new GammaDistribution[dim];
		for (int i = 0; i < dim; i++) {
			gammas[i] = new GammaDistribution((df-i+0.0)/2, 2);
		}
		CholeskyDecomposition cholesky = new CholeskyDecomposition(scaleMatrix);

		// Build N_{ij}
		double [][] N = new double[dim][dim];
		for (int j = 0; j < dim; j++) {
			for (int i = 0; i < j; i++) {
				N[i][j] = random.nextGaussian();
			}
		}

		// Build V_j
		double [] V = new double[dim];
		for (int i = 0; i < dim; i++) {
			V[i] = gammas[i].sample();
		}

		// Build B matrix
		double [][] B = new double[dim][dim];
		for (int j = 0; j < dim; j++) {
			double sum = 0.0;
			for (int i = 0; i < j; i++) {
				sum += Math.pow(N[i][j], 2);
			}
			B[j][j] = V[j] + sum;
		}
		for (int j = 1; j < dim; j++) {
			B[0][j] = N[0][j] * Math.sqrt(V[0]);
			B[j][0] = B[0][j];
		}
		for (int j = 1; j < dim; j++) {
			for (int i = 1; i < j; i++) {
				double sum = 0;
				for (int k = 0; k < i; k++) {
					sum += N[k][i] * N[k][j];
				}
				B[i][j] = N[i][j] * Math.sqrt(V[i]) + sum;
				B[j][i] = B[i][j];
			}
		}

		RealMatrix BMatrix = new Array2DRowRealMatrix(B);
		RealMatrix S = cholesky.getL().multiply(BMatrix).multiply(cholesky.getLT());
		S = IOUtils.convertMatrixToSymmetry(S);  // 如果矩阵不是对称矩阵,转化成对称矩阵

		return S;
	}
	/**sample from multivariate normal distribution
	 * give your mean and variance
	 * ****/
	public static double [] sampleFromMultivariateDistribution(double [] mean, double [][] variance){
		MultivariateNormalDistribution cc = new MultivariateNormalDistribution(mean, variance);
		double[] sampleValues = cc.sample();
		return sampleValues;
	}
	/**E-STEP 
	 * mean is the value that sampling from multivariate normal distribution
	 * spmpling_inverse_wishart is the value that sampling from inverse wishart distribution  
	 * variance is the our difine.
	 * ****/
	@SuppressWarnings("unused")
	private static Map<Integer,List<RealMatrix>> CalculateExpectaction(double [] mean,RealMatrix   spmpling_inverse_wishart,double variance,Map<Integer,RealMatrix > xmatrix,Map<Integer,RealMatrix > ymatrix) {
		Map<Integer,List<RealMatrix>> weMap=new HashMap<Integer,List<RealMatrix>>();
		for( int itemnumber : ymatrix.keySet() ){
			RealMatrix meanmatrix = new Array2DRowRealMatrix(mean);
			//计算逆矩阵
			RealMatrix inverse_inverse_wishartMatrix = inverseMatrix(spmpling_inverse_wishart);
			//计算x*x的和除以方差
			RealMatrix xtransposematrix=xmatrix.get(itemnumber).transpose();
			RealMatrix xxMatrix=xmatrix.get(itemnumber).preMultiply(xtransposematrix).scalarMultiply(1/variance);
			RealMatrix inverse_first=inverseMatrix(inverse_inverse_wishartMatrix.add(xxMatrix));
			RealMatrix second=xmatrix.get(itemnumber).transpose().multiply(ymatrix.get(itemnumber).transpose()).scalarMultiply(1/variance).add(inverse_inverse_wishartMatrix.multiply(meanmatrix));
			RealMatrix wMatrix=inverse_first.multiply(second);
			//下面对协方差矩阵更新进行计算
			RealMatrix covariancematrix=inverseMatrix(inverse_inverse_wishartMatrix.add(xxMatrix));
			System.out.println("covariancematrix:"+covariancematrix);
			List<RealMatrix> listRealMatrix=new ArrayList<RealMatrix>();
			//获取的w及协方差
			listRealMatrix.add(wMatrix);
			listRealMatrix.add(covariancematrix);
			weMap.put(itemnumber, listRealMatrix);
		}
		return weMap;
	}
	/**M-STEP 
	 * mean is the value that sampling from multivariate normal distribution
	 * spmpling_inverse_wishart is the value that sampling from inverse wishart distribution  
	 * variance is the our define.
	 * ****/
	@SuppressWarnings("unused")
	private static MStepData CalculateMaximization(Map<Integer,RealMatrix > wMatrixList,Map<Integer,RealMatrix > covariancematrixlist,Map<Integer,RealMatrix > xmatrix,Map<Integer,RealMatrix > ymatrix) {
		//uw update
		RealMatrix meanupdate= updatew(wMatrixList).scalarMultiply(1/Double.valueOf(wMatrixList.size()));
		//covariance update 
		RealMatrix covarianceupdate=updatecovariance(wMatrixList,covariancematrixlist,meanupdate).scalarMultiply(1/Double.valueOf(wMatrixList.size()));
		//variance update 
		double varianceupdate=updatevariance(wMatrixList,xmatrix,ymatrix);
		List<RealMatrix> listRealMatrix=new ArrayList<RealMatrix>();
		MStepData datamodel=new MStepData();
		datamodel.setMeanupdate(meanupdate);
		datamodel.setCovarianceupdate(covarianceupdate);
		datamodel.setVarianceupdate(varianceupdate);
		return datamodel;
	}
	//update uw均值
	private static RealMatrix updatew(Map<Integer,RealMatrix > wMatrixList){
		//suMatrix会报错
		Integer[] keys = wMatrixList.keySet().toArray(new Integer[0]);
		//然後随机一个键，找出该值
		Random random = new Random();
		System.out.println("wMatrixList:\t"+wMatrixList+"\tkey-random:\t"+random.nextInt(keys.length));
		Integer randomKey = keys[random.nextInt(keys.length)];
		RealMatrix randomMatrix = wMatrixList.get(randomKey);
		RealMatrix suMatrix= new Array2DRowRealMatrix(randomMatrix.getRowDimension(),1);
		for( Entry<Integer,RealMatrix> entry : wMatrixList.entrySet() ){
			suMatrix=suMatrix.add(entry.getValue());
		}
		return suMatrix;
	}
	//update covariance 
	private static RealMatrix updatecovariance(Map<Integer,RealMatrix > wMatrixList,Map<Integer,RealMatrix > covariancematrix,RealMatrix meanmatrixupdate){
		Integer[] keys = wMatrixList.keySet().toArray(new Integer[0]);
		//然後随机一个键，找出该值
		Random random = new Random();
		Integer randomKey = keys[random.nextInt(keys.length)];
		RealMatrix randomMatrix = wMatrixList.get(randomKey);
		RealMatrix suMatrix= new Array2DRowRealMatrix(randomMatrix.getRowDimension(),randomMatrix.getRowDimension());
		for( int itemnumber : wMatrixList.keySet() ){
			RealMatrix aMatrix=wMatrixList.get(itemnumber).add(meanmatrixupdate.scalarMultiply(-1.0));
			RealMatrix bMatrix=aMatrix.transpose();
			RealMatrix cMatrix=aMatrix.multiply(bMatrix);
			suMatrix=suMatrix.add(covariancematrix.get(itemnumber).add(cMatrix));
		}
		return suMatrix;
	}
	/**update variance,a double vaule 
	 * ymatrix contains item and score
	 * ****/

	@SuppressWarnings("unused")
	private static double updatevariance(Map<Integer,RealMatrix > wMatrixList,Map<Integer,RealMatrix > xmatrix,Map<Integer,RealMatrix > ymatrix){
		//y-wx square
		double sum=0.0;
		int dimension=0;
		for( int itemnumber : ymatrix.keySet() ){
			RealMatrix yscore=ymatrix.get(itemnumber) ;
			dimension+=yscore.getColumnDimension();
			RealMatrix wxscore=wMatrixList.get(itemnumber).transpose().multiply(xmatrix.get(itemnumber).transpose()) ;
			RealMatrix ysbuwxscore =yscore.add(wxscore.scalarMultiply(-1.0));
			sum+=sumarray(ysbuwxscore.multiply(ysbuwxscore.transpose()));
		}
		return sum/dimension;
	}
	//求矩阵所有元素之和
	private static double sumarray(RealMatrix a){
		double[][] arr=a.getData();
		double sum = 0.0; 
		for (int i = 0; i < arr.length; i++) {  
			for (int j = 0; j < arr[i].length; j++) {  
				sum += arr[i][j];  
			}  
		} 
		return sum;
	}
	//字符型数组转化为double型数组
	private static double[] StringtoDouble(String []a){

		double[] ds=new double[a.length];
		for(int i=0;i<a.length;i++){
			ds[i]=Double.valueOf(a[i]);
		}
		return ds;
	}
	//将xmap形式list转化为矩阵
	private static Hashtable<Integer, RealMatrix> StringarrtoDouble(Hashtable<Integer,List<double[]>> maplist){
		Hashtable<Integer,RealMatrix> maparr=new Hashtable<Integer,RealMatrix>();
		for (HashMap.Entry<Integer, List<double[]>> entry : maplist.entrySet()) {
			double[][] arr=new double[entry.getValue().size()][];
			for (int i = 0; i < entry.getValue().size(); i++) {
				arr[i]=entry.getValue().get(i);
			}
			RealMatrix xvalue = new Array2DRowRealMatrix(arr);
			maparr.put(entry.getKey(), xvalue);
		}
		return maparr;
	}
	//将ymap形式list转化为矩阵
	private static Hashtable<Integer, RealMatrix> y_arrtoDouble(Hashtable<Integer,List<Double>> maplist){
		Hashtable<Integer,RealMatrix> maparr=new Hashtable<Integer,RealMatrix>();
		for (HashMap.Entry<Integer, List<Double>> entry : maplist.entrySet()) {
			double[][] arr=new double[1][entry.getValue().size()];
			for (int i = 0; i < entry.getValue().size(); i++) {
				arr[0][i]=entry.getValue().get(i);
			}
			RealMatrix xvalue = new Array2DRowRealMatrix(arr);
			maparr.put(entry.getKey(), xvalue);
		}
		return maparr;
	}
}


