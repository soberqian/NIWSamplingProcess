package niwsampling;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import java.io.OutputStreamWriter;

public class IOUtils {
	public static double[] cc = { 76.18009172947146D, -86.50532032941678D, 24.01409824083091D, -1.231739572450155D, 0.001208650973866179D, -5.395239384953E-6D };
	
	public static BufferedReader getBufferedReader(String filepath) throws FileNotFoundException, UnsupportedEncodingException{
		BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(filepath), "UTF-8"));
	    return in;
	}
		
	
	public static BufferedWriter getBufferedWriter(String filepath) throws FileNotFoundException, UnsupportedEncodingException{
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filepath), "UTF-8"));
	    return out;
	}  
	
	
	/***sample the eta value
	 * 主要参考论文:Scalable Inference for Logistic-Normal Topic Models
	 * @param  i            需要给i标签采样eta参数
	 * @param  mean         多元正态分布的均值向量
	 * @param  inverseSigma 多元正态分布的协方差的逆矩阵中第labelIndex行对应的数组
	 * @param  eta          当前的该文档对应的eta数组
	 * @param  lambda       该文档中第i个标签对应lambda参数
	 * @param  labelNumber  表示该文档中属于标签labelIndex的单词的数量
	 * @param  totalNumber  表示该文档中共有的单词的数量
	 * ****/
	public static double sampleEta (int i, double [] mean, double [] inverseSigma, double [] eta, double lambda_d_i, int labelNumber, double totalNumber) {
		double newEta = 0.0;
		// 计算mu_d_i
		double mu_i = mean[i] - (new ArrayRealVector(inverseSigma).dotProduct(new ArrayRealVector(eta).subtract(new ArrayRealVector(mean))) 
				- inverseSigma[i] * (eta[i] - mean[i])) / inverseSigma[i];
		// 计算方差
		double tau = 1.0 / (inverseSigma[i] + lambda_d_i);
		// 计算后验均值
		double gamma = tau * (mu_i * inverseSigma[i] + labelNumber - 0.5 * totalNumber + lambda_d_i * IOUtils.LogSumExpValue(eta, i));
		newEta = IOUtils.sampleFromUnivariateNormalDistribution(gamma, tau);
		return newEta;
	}
	
	/**sample from inverse wishart distribution  
	 * 采样威沙特分布实现的，而威沙特分布的尺度矩阵其实是逆威沙特分布的逆矩阵，所以下面中有一句inverseMatrix(scaleMatrix)。
	 * 目前只有C#代码https://github.com/mathnet/mathnet-numerics/blob/master/src/Numerics/Distributions/InverseWishart.cs 明确涉及到这一点。
	 * @param  df  逆威沙特分布的自由度
	 * @param  scaleMatrix  逆威沙特分布的尺度矩阵***/
	public static RealMatrix sampleFromInverseWishartDistribution(double df, RealMatrix scaleMatrix){
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
	
	/**sample from wishart distribution
	 * 采样威沙特分布实现的，而威沙特分布的尺度矩阵其实是逆威沙特分布的逆矩阵
	 * @param   df   威沙特分布的自由度
	 * @param   scaleMatrix   威沙特分布的尺度矩阵       它与逆威沙特分布的尺度矩阵是互逆的关系****/
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
	 * 从正太分布中采样
	 * ****/
	public static double [] sampleFromMultivariateDistribution(double [] mean, double [][] variance){
		MultivariateNormalDistribution cc = new MultivariateNormalDistribution(mean, variance);
		double[] sampleValues = cc.sample();
		return sampleValues;
	}
	
	/***
	 * 参考文章：Scalable Inference for Logistic-Normal Topic Models
	 * 根据中心极限定理,同n很大时,可以近似通过正态分布从PG中进行采样.
	 * 同时,当PG分布的第二参数为0时，它的均值和方差可以通过求极限得到(洛必达法则，具体参考博客)。
	 * 对于分子和分母都为0的极限是通过洛必达法则实现.
	 * 这里lambda服从PG(N,rho)分布
	 * @param  n  
	 * @param  rho****/
	public static double sampleLambda(int n, double rho) {
		double newLambda = 0.0;
		double mu = 0.0, variance = 0.0;
		// 如果rho参数为0,则根据求极限得到此时的均值和方差（洛必达法则）
		if (rho == 0.0) {
			mu = (n + 0.0) / 4.0;
			variance = (n + 0.0) / 24.0;
			System.out.println("mu == 0");
		} else {
			mu = 0.5 * (n + 0.0) / rho * Math.tanh(rho / 2.0);
			double mu2 = (Math.pow(rho, 2) * n * Math.cosh(rho) + 2.0 * rho * Math.sinh(rho) - 
					(2.0 + n) * Math.pow(rho, 2)) * n / 8.0 / Math.pow(rho, 4) / Math.pow(Math.cosh(rho/2.0), 2);
			variance = mu2 - mu * mu;
		}
		newLambda = IOUtils.sampleFromUnivariateNormalDistribution(mu, variance);;
		return newLambda;
	}
	
	/**sample from univariate normal distribution****/
	public static double sampleFromUnivariateNormalDistribution (double mean, double variance) {
		NormalDistribution norDis = new NormalDistribution(mean, variance);
		double randomValue = norDis.sample();
		return randomValue;
	}
	/**求逆矩阵***/
	public static RealMatrix inverseMatrix(RealMatrix A) {
        RealMatrix result = new LUDecomposition(A).getSolver().getInverse();
        return result; 
    }
	
	/**这个方法是用来计算log(\Sigma_{i\neq k}e^{arrry[i])
	 * 主要这里的i不等于k***/
	public static double LogSumExpValue (double [] array, int k) {
		double returnvalue = 0.0;
		for (int i = 0; i < array.length; i++) {
			if (i != k) {
				returnvalue += Math.exp(array[i]);
			}
		}
		returnvalue = Math.log(returnvalue);
		return returnvalue;
	}
	
	/***轮盘赌****/
	public static int scaleSample(double[] distribution){
		double[] cumm_probs = new double[distribution.length];
		System.arraycopy(distribution, 0, cumm_probs, 0, cumm_probs.length);
		for (int i = 1; i < cumm_probs.length; i++) {
			cumm_probs[i] += cumm_probs[(i - 1)];
		}
		double randValue = Math.random() * cumm_probs[(cumm_probs.length - 1)];
		int index;
		for (index = 0; index < cumm_probs.length; index++) {
			if (randValue < cumm_probs[index]) {
				break;
			}
		}
		return index;
	}
	
	public static double logGammaStirling(double x) {
		if (x == 0.0D) {
			return logGammaStirling(1.0E-300D);
		}
		if (x < 1.0E-300D) {
			return logGammaDefinition(x);
		}
		
		double y = x;
		double t = x + 5.5D;
		t -= (x + 0.5D) * Math.log(t);
		double r = 1.000000000190015D;
		for (int i = 0; i < 6; i++) {
			r += cc[i] / ++y;
		}
		return -t + Math.log(2.5066282746310007D * r / x);
	}
	
	public static double logGammaDefinition(double z) {
		double result = -0.5772156649015329D * z - Math.log(z);
		for (int k = 1; k < 10000000; k++) {
			result += z / k - Math.log(1.0D + z / k);
		}
		return result;
	}
	
	// 计算阶乘的log值
    public static double sumLogFactorial (int number) {
    	double samplevalue = 0.0;
    	for (int n = 1; n < number; n++) {
    		samplevalue += Math.log(n);
		}
    	return samplevalue;
    }
    
    public static List rankDoubleList (Set<Map.Entry<Integer,Double>> s) {
    	List<Map.Entry<Integer, Double>> list = new ArrayList<Map.Entry<Integer, Double>>(s);
    	Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>() {  
            //降序排序  
            public int compare(Entry<Integer, Double> o1, Entry<Integer, Double> o2) {  
                //return o1.getValue().compareTo(o2.getValue());  
                return o2.getValue().compareTo(o1.getValue());  
            }  
        });
    	return list;
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
    
}
