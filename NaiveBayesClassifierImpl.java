import java.util.HashMap;
import java.lang.Math;

/**
 * Implementation of a naive bayes classifier.
 */
public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {
	private Instance[] m_trainingData;
	private int m_v;
	private double m_delta;
	public int m_sports_count, m_business_count;
	public int m_sports_word_count, m_business_word_count;
	private HashMap<String,Integer> m_map[] = new HashMap[2];

  /**
   * Trains the classifier with the provided training data and vocabulary size
   */
  @Override
  public void train(Instance[] trainingData, int v) {
    // For all the words in the documents, count the number of occurrences. Save in HashMap
    // e.g.
    // m_map[0].get("catch") should return the number of "catch" es, in the documents labeled sports
    // m_map[0].get("asdasd") would return null, when the word has not appeared before.
    // m_map[0].put(word,1) puts the first count in.
    // m_map[0].replace(word, count+1) updates the value
  	  m_trainingData = trainingData;
  	  m_v = v;
  	  m_map[0] = new HashMap<>();
  	  m_map[1] = new HashMap<>();
  	  for (int i = 0; i < m_trainingData.length; i++)
  		  if (trainingData[i].label == Label.SPORTS)
  			  for (int j = 0; j <trainingData[i].words.length; j++)
  				  if (m_map[0].get(trainingData[i].words[j]) == null)
  					  m_map[0].put(trainingData[i].words[j], 1);
  				  else
  					  m_map[0].put(trainingData[i].words[j], m_map[0].get(trainingData[i].words[j]) + 1);
  		  else
  			  for (int j = 0; j <trainingData[i].words.length; j++)
  				  if (m_map[1].get(trainingData[i].words[j]) == null)
					  m_map[1].put(trainingData[i].words[j], 1);
  				  else
  					  m_map[1].put(trainingData[i].words[j], m_map[1].get(trainingData[i].words[j]) + 1);
  }

  /**
   * Counts the number of documents for each label
   */
  public void documents_per_label_count(Instance[] trainingData){
    m_sports_count = 0;
    m_business_count = 0;
    for (int i = 0; i < trainingData.length; i++)
		if (trainingData[i].label == Label.SPORTS)
			m_sports_count++;
		else
			m_business_count++;
  }

  /**
   * Prints the number of documents for each label
   */
  public void print_documents_per_label_count(){
  	  System.out.println("SPORTS=" + m_sports_count);
  	  System.out.println("BUSINESS=" + m_business_count);
  }

  /**
   * Counts the total number of words for each label
   */
  public void words_per_label_count(Instance[] trainingData){
    m_sports_word_count = 0;
    m_business_word_count = 0;
    for (int i = 0; i < trainingData.length; i++)
		if (trainingData[i].label == Label.SPORTS)
			m_sports_word_count += trainingData[i].words.length;
		else
			m_business_word_count += trainingData[i].words.length;
  }

  /**
   * Prints out the number of words for each label
   */
  public void print_words_per_label_count(){
  	  System.out.println("SPORTS=" + m_sports_word_count);
  	  System.out.println("BUSINESS=" + m_business_word_count);
  }

  /**
   * Returns the prior probability of the label parameter, i.e. P(SPORTS) or P(BUSINESS)
   */
  @Override
  public double p_l(Label label) {
    // Calculate the probability for the label. No smoothing here.
    // Just the number of label counts divided by the number of documents.
    int sportsNum = 0, businessNum = 0;
	double ret = 0;
    for (int i = 0; i < m_trainingData.length; i++)
    	if (label == Label.SPORTS)
    		if (m_trainingData[i].label == label)
    			sportsNum++;
    		else
    			businessNum++;
    	else
    		if (m_trainingData[i].label == label)
    			businessNum++;
    		else
    			sportsNum++;
    if (label == Label.SPORTS)
    	ret = sportsNum * 1.0 / m_trainingData.length;
    else
    	ret = businessNum * 1.0 / m_trainingData.length;
    return ret;
  }

  /**
   * Returns the smoothed conditional probability of the word given the label, i.e. P(word|SPORTS) or
   * P(word|BUSINESS)
   */
  @Override
  public double p_w_given_l(String word, Label label) {
    // Calculate the probability with Laplace smoothing for word in class (label)
    double ret = 0;
    m_delta = 0.00001;
    words_per_label_count(m_trainingData);
    if (label == Label.SPORTS)
    	if (m_map[0].get(word) == null)
    		ret = m_delta / (m_v * m_delta + m_sports_word_count);
    	else
    		ret = (m_map[0].get(word) + m_delta) / (m_v * m_delta + m_sports_word_count);
    else
    	if (m_map[1].get(word) == null)
    		ret = m_delta / (m_v * m_delta + m_business_word_count);
    	else
    		ret = (m_map[1].get(word) + m_delta) / (m_v * m_delta + m_business_word_count);
    return ret;
  }

  /**
   * Classifies an array of words as either SPORTS or BUSINESS.
   */
  @Override
  public ClassifyResult classify(String[] words) {
    // Sum up the log probabilities for each word in the input data, and the probability of the label
    // Set the label to the class with larger log probability
    ClassifyResult ret = new ClassifyResult();
    ret.label = Label.SPORTS;
    ret.log_prob_sports = 0;
    ret.log_prob_business = 0;
    double sum_sports = 0, sum_business = 0;
    for (int k = 0; k < words.length; k++) {
    	sum_sports += Math.log(p_w_given_l(words[k], Label.SPORTS));
    	sum_business += Math.log(p_w_given_l(words[k], Label.BUSINESS));
    }
    ret.log_prob_sports = Math.log(p_l(Label.SPORTS)) + sum_sports;
    ret.log_prob_business = Math.log(p_l(Label.BUSINESS)) + sum_business;
    ret.label = (ret.log_prob_sports > ret.log_prob_business)? Label.SPORTS : Label.BUSINESS;
    return ret; 
  }
  
  /**
   * Constructs the confusion matrix
   */
  @Override
  public ConfusionMatrix calculate_confusion_matrix(Instance[] testData){
    // Count the true positives, true negatives, false positives, false negatives
    int TP, FP, FN, TN;
    TP = 0;
    FP = 0;
    FN = 0;
    TN = 0;
    for (int i = 0; i < testData.length; i++)
    	if (testData[i].label == Label.SPORTS)
    		if (classify(testData[i].words).label == Label.SPORTS)
    			TP++;
    		else
    			FN++;
    	else
    		if (classify(testData[i].words).label == Label.BUSINESS)
    			TN++;
    		else
    			FP++;
    return new ConfusionMatrix(TP,FP,FN,TN);
  } 
}