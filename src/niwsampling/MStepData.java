package niwsampling;

import org.apache.commons.math3.linear.RealMatrix;

public class MStepData {
	private RealMatrix meanupdate;
	private RealMatrix covarianceupdate;
	private double varianceupdate;
	private RealMatrix wMatrixList;
	
	public RealMatrix getwMatrixList() {
		return wMatrixList;
	}
	public void setwMatrixList(RealMatrix wMatrixList) {
		this.wMatrixList = wMatrixList;
	}
	public RealMatrix getMeanupdate() {
		return meanupdate;
	}
	public void setMeanupdate(RealMatrix meanupdate) {
		this.meanupdate = meanupdate;
	}
	public RealMatrix getCovarianceupdate() {
		return covarianceupdate;
	}
	public void setCovarianceupdate(RealMatrix covarianceupdate) {
		this.covarianceupdate = covarianceupdate;
	}
	public double getVarianceupdate() {
		return varianceupdate;
	}
	public void setVarianceupdate(double varianceupdate) {
		this.varianceupdate = varianceupdate;
	}
}
