# UI - Zadanie 4
## Rozpoznávanie vzorov pomocou strojového učenia

Najkvalitnejšie modely boli backpropagation a Random Forest, ktoré mali podobnú presnosť (0.96 a 0.94), pričom rozhodovacie stromy na tom boli horšie (0.86).
```
Classification report for classifier MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=100, learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='adam', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False):
              precision    recall  f1-score   support

         0.0       0.98      0.98      0.98      3475
         1.0       0.99      0.98      0.98      3925
         2.0       0.96      0.95      0.95      3524
         3.0       0.94      0.94      0.94      3631
         4.0       0.98      0.93      0.95      3404
         5.0       0.95      0.95      0.95      3143
         6.0       0.96      0.98      0.97      3410
         7.0       0.95      0.97      0.96      3653
         8.0       0.94      0.94      0.94      3389
         9.0       0.93      0.95      0.94      3446

   micro avg       0.96      0.96      0.96     35000
   macro avg       0.96      0.96      0.96     35000
weighted avg       0.96      0.96      0.96     35000


Confusion matrix:
[[3410    0    7    0    2   10   13    7   19    7]
 [   1 3841   23    8    9    2   10   10   19    2]
 [  18   11 3336   46    7   12   15   37   34    8]
 [   8    7   42 3429    4   72    1   24   27   17]
 [   5    4   26    1 3151    2   22   31   30  132]
 [   8    1    3   52    4 2981   32    5   33   24]
 [  14    5    1    0    8   16 3353    1   12    0]
 [   3    9   20   28    6    3    3 3557    4   20]
 [  16   11   10   55    6   32   29   22 3188   20]
 [   6    4    1   34   22   18    4   64   22 3271]]
```
```
Classification report for classifier DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'):
              precision    recall  f1-score   support

         0.0       0.92      0.91      0.91      3475
         1.0       0.94      0.94      0.94      3925
         2.0       0.83      0.85      0.84      3524
         3.0       0.83      0.82      0.83      3631
         4.0       0.85      0.85      0.85      3404
         5.0       0.80      0.81      0.81      3143
         6.0       0.88      0.88      0.88      3410
         7.0       0.89      0.89      0.89      3653
         8.0       0.79      0.79      0.79      3389
         9.0       0.82      0.82      0.82      3446

   micro avg       0.86      0.86      0.86     35000
   macro avg       0.86      0.86      0.86     35000
weighted avg       0.86      0.86      0.86     35000


Confusion matrix:
[[3150    2   40   30   20   70   54   14   59   36]
 [   3 3709   48   32   16   26   15   18   47   11]
 [  39   38 2999   97   42   41   62   76   89   41]
 [  39   34  116 2973   32  171   36   53   97   80]
 [  23   15   56   21 2909   30   55   57   77  161]
 [  49   29   40  153   46 2544   84   37   89   72]
 [  48   17   73   28   75   67 2994   12   79   17]
 [   7   32   76   57   46   34   19 3242   29  111]
 [  39   60  109  111   70  128   57   39 2674  102]
 [  23   22   41   71  160   61   18   99  131 2820]]
```
```
Classification report for classifier RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False):
              precision    recall  f1-score   support

         0.0       0.96      0.98      0.97      3475
         1.0       0.97      0.98      0.98      3925
         2.0       0.92      0.94      0.93      3524
         3.0       0.91      0.92      0.92      3631
         4.0       0.92      0.95      0.94      3404
         5.0       0.92      0.91      0.92      3143
         6.0       0.96      0.96      0.96      3410
         7.0       0.95      0.94      0.95      3653
         8.0       0.93      0.89      0.91      3389
         9.0       0.93      0.90      0.92      3446

   micro avg       0.94      0.94      0.94     35000
   macro avg       0.94      0.94      0.94     35000
weighted avg       0.94      0.94      0.94     35000


Confusion matrix:
[[3397    1    7   11    5   11   19    4   17    3]
 [   2 3854   27   12    4    5    3    9    6    3]
 [  25   10 3322   35   26   13   17   40   29    7]
 [  12   10   77 3349    5   70    6   41   44   17]
 [  14    4   19    6 3231    3   18   11   20   78]
 [  21   14   10  105   21 2874   30    7   37   24]
 [  19    9   16    8   24   50 3272    1   10    1]
 [  10   14   68   19   34    1    1 3444   16   46]
 [  17   33   53   71   40   72   25    4 3032   42]
 [  17   11   14   58  116   25    4   52   52 3097]]
```

Pri algoritme backpropagation bolo najväčšie zlepšenie zvýšenim počtu skrytých vrstiev neurónovej siete (zo 100 na 1000) kde sa zvýšila presnosť o 0.01.

Pri algoritme rozhodovacích stromov malo najvačší vplyv na presnosť zmena maximálnej hĺbky stromu.

Pri algoritme Random Forest sa presnosť zvýšila o 0.02 pri zvýšení počtu estimatorov z 10 na 100.

Po skombinovaní všetkých troch modelov stackovaním, kde bol použitý meta klasifikátor logistickej regresie, sa oproti najlepšiemu algoritmu znížila presnosť o 0.03.

```
Classification report for classifier StackingClassifier(average_probas=False,
          classifiers=[MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=100, learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       n_iter_no_change=10, neste...jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)],
          meta_classifier=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False),
          store_train_meta_features=False, use_clones=True,
          use_features_in_secondary=False, use_probas=True, verbose=0):
              precision    recall  f1-score   support

         0.0       0.96      0.96      0.96      3475
         1.0       0.97      0.98      0.97      3925
         2.0       0.91      0.92      0.92      3524
         3.0       0.90      0.89      0.90      3631
         4.0       0.92      0.91      0.92      3404
         5.0       0.90      0.90      0.90      3143
         6.0       0.95      0.96      0.95      3410
         7.0       0.95      0.94      0.95      3653
         8.0       0.89      0.89      0.89      3389
         9.0       0.89      0.89      0.89      3446

   micro avg       0.93      0.93      0.93     35000
   macro avg       0.92      0.92      0.92     35000
weighted avg       0.93      0.93      0.93     35000


Confusion matrix:
[[3341    3   17   18    7   23   16    6   31   13]
 [   1 3831   26   12    5    8    7   10   18    7]
 [  29    9 3243   52   21   18   35   43   45   29]
 [  16    8   90 3238   15   98   16   30   65   55]
 [   6    6   29   10 3109   20   25   21   45  133]
 [  18   17   13  109   17 2821   48    8   63   29]
 [  29   11   24    6   19   28 3257    2   25    9]
 [   9   21   31   41   22    5    4 3441   16   63]
 [  17   27   47   69   37   64   28   15 3033   52]
 [  13    7   31   44  125   35    6   45   77 3063]]
```

Najväčší vplyv na kvalitu kombinovaného modelu má kombinácia použitých modelov, pri odobratí modelu s najnižšsou presnosťou (rozhodovacie stromy) sa zvýšila presnosť kombinovaného modelu na 0.97.