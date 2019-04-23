/**
 * https://ybigta-data-science.readthedocs.io/en/latest/6_Data_Science_from_Scratch/02_Gradient%20Descent%20&%20Linear%20Regression/
 * 예제의 파이선코드를 자바로 바꾸면서 이론 공부
 * 
 * @author 김진용
 *
 */
public class GradientDescentExample {
	public static void main(String args[]) {
		double[] num_friends_good = {49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
		double[] daily_minutes_good = {68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84};
		
		GradientDescentExample example = new GradientDescentExample();
		
		double[] arr = example.makeArr(10);
		
		Expression expression = example.new Expression();
		
		System.out.println(expression.toString(arr));
		
		for(int i = 0; i < arr.length; i++) {
			System.out.println(example.partialDifferenceQuotient(expression, arr, i, 1e-10));
		}
		
		double[] result = example.getLeastSquaresFit(num_friends_good, daily_minutes_good);
		
		for(int i = 0; i < result.length; i++)
			System.out.println(result[i]);
	}
	
	public double[] makeArr(int length) {
		double[] arr = new double[length];
		for(int i = 0; i < length; i++) {
			arr[i] = i;
		}
		
		return arr;
	}
	
	/**
	 * @param alpha
	 * @param beta
	 * @param x1
	 * @return beta * x1 + alpha
	 */
	public double predict(double alpha, double beta, double x1) {
		return (beta * x1) + alpha;
	}
	
	/**
	 * 오차를 확인하는 메소드
	 * @param alpha
	 * @param beta
	 * @param x1
	 * @param y1
	 * @return
	 */
	public double getError(double alpha, double beta, double x1, double y1) {
		return y1 - predict(alpha, beta, x1);
	}
	
	/**
	 * 최소제곱법을 이용해 beta * x + alpha 의 alpha, beta값을 구하는 메소드
	 * 
	 * @param x
	 * @param y
	 * @return result[0] = beta, result[1] = alpha
	 */
	public double[] getLeastSquaresFit(double[] x, double[] y) {
		double[] result = new double[2];
		
		result[1] = correlation(x, y) * std(y) / std(x);
		result[0] = mean(y) - result[1] * mean(x);
		
		return result;
	}
	
	/**
	 * 상관 분석값 알아내는 메소드, 두개의 정보에 대한 상관 분석값을 계산하여 값을 리턴해준다.
	 * <li>return이 -1.0과 -0.7 사이이면, 강한 음적 선형관계,</li>
	 * <li>return이 -0.7과 -0.3 사이이면, 뚜렷한 음적 선형관계,</li>
	 * <li>return이 -0.3과 -0.1 사이이면, 약한 음적 선형관계,</li>
	 * <li>return이 -0.1과 +0.1 사이이면, 거의 무시될 수 있는 선형관계,</li>
	 * <li>return이 +0.1과 +0.3 사이이면, 약한 양적 선형관계,</li>
	 * <li>return이 +0.3과 +0.7 사이이면, 뚜렷한 양적 선형관계,</li>
	 * <li>return이 +0.7과 +1.0 사이이면, 강한 양적 선형관계</li>
	 * 
	 * @param arr1
	 * @param arr2
	 * @return correlation
	 */
	private double correlation(double[] arr1, double[] arr2) {
		
		double mean1 = mean(arr1);
		double mean2 = mean(arr2);
		
		double numer = 0.0;
		
		double temp1 = 0.0;
		double temp2 = 0.0;
		
		for(int i = 0; i < arr1.length; i++) {
			temp1 += Math.pow(arr1[i] - mean1, 2);
			temp2 += Math.pow(arr2[i] - mean2, 2);
		}
		
		for(int i = 0; i < arr1.length; i++) {
			numer += (arr1[i] - mean1) * (arr2[i] - mean2);
		}
		
		return numer / Math.sqrt(temp1 * temp2);
	}
	
	public double sumOfSquares(double[] arr) {
		double sum = 0.0;
		
		for(int i = 0; i < arr.length; i++) {
			sum += Math.pow(arr[i], 2);
		}
		
		return sum;
	}
	
	/**
	 * 목적함수(expression)에서 변화율 h 가 들어왔을 때 변화량을 측정하는 메소드
	 * 단일 변화량을 구함
	 * 
	 * @param expression
	 * @param arr
	 * @param h
	 * @return 변화량 / h
	 */
	public double difference(Expression expression, double[] arr, double h) {
		double[] tempArr = new double[arr.length];
		
		for(int i = 0; i < arr.length; i++) {
			tempArr[i] = arr[i] + h;
		}
		
		return (expression.getPredictSum(expression.getPredictArr(tempArr)) - expression.getPredictSum(expression.getPredictArr(arr))) / h;
	}
	
	public double partialDifferenceQuotient(Expression expression, double[] arr, int index, double h) {
		double[] tempArr = new double[arr.length];
		
		for(int i = 0; i < arr.length; i++) {
			if(i == index) {
				tempArr[i] = arr[i] + h;
			}
			else {
				tempArr[i] = arr[i];
			}
				
		}
		
		return (expression.getPredictSum(expression.getPredictArr(tempArr)) - expression.getPredictSum(expression.getPredictArr(arr))) / h;
	}
	
	/**
	 * 표준편차 구하는 메소드
	 * <li>sqrt( sum( (변량 - 평균)^2) / 변량개수 )</li>
	 * 
	 * @param arr
	 * @return standard_deviation
	 */
	public double std(double[] arr) {
		if(arr == null || arr.length == 0)
			return 0.0;
		
		double mean = mean(arr);
		double sum = 0.0;
		
		for(int i = 0; i < arr.length; i++) {
			sum += Math.pow(arr[i] - mean, 2);
		}
		
		return Math.sqrt(sum / arr.length);
	}
	
	/**
	 * 평균값
	 * @param arr
	 * @return mean
	 */
	public double mean(double[] arr) {
		if(arr == null || arr.length == 0)
			return 0.0;
		
		double sum = 0.0;
		for(int i = 0; i < arr.length; i++) {
			sum+= arr[i];
		}
		
		return sum / arr.length;
	}
	
	
	/**
	 * 목적함수 정의 클래스
	 * @author 김진용
	 *
	 */
	public class Expression {
		
		public double getPredictSquaredSum(double[] arr) {
			double sum = 0.0;
			
			for(int i = 0; i < arr.length; i++) {
				sum = Math.pow(arr[i], 2);
			}
			
			return sum;
		}
		
		public double getPredictSum(double[] arr) {
			double sum = 0.0;
			
			for(int i = 0; i < arr.length; i++) {
				sum += arr[i];
			}
			
			return sum;
		}
		
		public double[] getPredictArr(double[] arr) {
			double[] result = new double[arr.length];
			
			for(int i = 0; i < arr.length; i++) {
				result[i] = i * arr[i];
			}
			
			return result;
		}
		
		
		/**
		 * y = ax + b
		 * @param alpha
		 * @param beta
		 * @param x
		 * @return y
		 */
		public double getPredict(double alpha, double beta, double x) {
			return alpha * x + beta;
		}
		
		public String toString(double[] arr) {
			if(arr.length < 1)
				return "";
			
			arr = getPredictArr(arr);
			
			StringBuilder sb = new StringBuilder();
			sb.append("getPredictArr=[");
			for(int i = 0; i < arr.length; i++) {
				if(i > 0)
					sb.append(", ");
				sb.append(arr[i]);
			}
			
			sb.append("]\n");
			sb.append("getPredictSum=").append(getPredictSum(arr));
			
			return sb.toString();
		}
	}
}
