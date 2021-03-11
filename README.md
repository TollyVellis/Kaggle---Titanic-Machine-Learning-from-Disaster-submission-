# Kaggle---Titanic-Machine-Learning-from-Disaster-submission-

This was my first stab at a Kaggle competition, on the famous Titanic dataset. It's basis lies in predicting whether a set of passengers whose fates you do not know either died or survived, based off applying ML to a separate data set of passengers who's fates you do know.

I'd say the biggest part was the data cleaning at the start. Firstly, reconfiguring the data to either remove irrelevant information (passenger names), turning qualitative features into quantitative (like male/female into 1/0) and most difficult of all, filling in missing information. This was mostly an issue for 'Age'; rather than use the median of all other ages, I hit on the idea to split passengers up by honorifics (Mrs, Mr, Miss/Ms and Master) and use medians by category, which I suspect gave much more accurate estimates. There was a few more columns which would be dropped, but this would happen later after further investigation.

I ultimately settled on doing the project in Python using a tuned SVM algorithm (SCM C=0.5, kernel='rbf') on standardised data, which gave an accuracy value of 0.77990 on Kaggle, making me 4550th on the leader board upon submission out of about 17500 entries, which apparently is quite good, especially on the first go (according to this thread anyway: https://www.kaggle.com/c/titanic/discussion/26284).
