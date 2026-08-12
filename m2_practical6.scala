import breeze.linalg.DenseVector
import breeze.numerics.sigmoid
import breeze.optimize.{DiffFunction, LBFGS}
import breeze.plot._

import scala.io.Source

object m2_practical6 {

  def main(args: Array[String]): Unit = {

    // ==========================================================
    // LOGISTIC REGRESSION USING BREEZE
    // Dataset: Pima Indians Diabetes Dataset
    // ==========================================================

    // ----------------------------------------------------------
    // 1. READ CSV FILE
    // ----------------------------------------------------------

    val source = Source.fromFile("diabetes.csv")

    val data =
      source
        .getLines()
        .drop(1)
        .toArray
        .map { line =>

          val v =
            line
              .split(",")
              .map(_.trim.replace("\"", ""))

          val features =
            DenseVector(
              v(0).toDouble, // Pregnancies
              v(1).toDouble, // Glucose
              v(2).toDouble, // BloodPressure
              v(3).toDouble, // SkinThickness
              v(4).toDouble, // Insulin
              v(5).toDouble, // BMI
              v(6).toDouble, // DiabetesPedigreeFunction
              v(7).toDouble  // Age
            )

          val label =
            v(8).toDouble

          (features, label)
        }

    source.close()

    println("==============================================")
    println("       LOGISTIC REGRESSION USING BREEZE")
    println("==============================================")

    println("Dataset loaded successfully.")
    println("Total records: " + data.length)

    // ----------------------------------------------------------
    // 2. TRAINING DATA
    // ----------------------------------------------------------

    val trainData =
      data.zipWithIndex
        .filter {
          case (_, i) => i % 5 != 0
        }
        .map(_._1)

    val numFeatures = 8

    // ----------------------------------------------------------
    // 3. CALCULATE MEAN
    // ----------------------------------------------------------

    val means =
      DenseVector.zeros[Double](numFeatures)

    for (j <- 0 until numFeatures) {

      var sum = 0.0

      for (i <- trainData.indices) {
        sum += trainData(i)._1(j)
      }

      means(j) =
        sum / trainData.length
    }

    // ----------------------------------------------------------
    // 4. CALCULATE STANDARD DEVIATION
    // ----------------------------------------------------------

    val stds =
      DenseVector.zeros[Double](numFeatures)

    for (j <- 0 until numFeatures) {

      var sum = 0.0

      for (i <- trainData.indices) {

        val difference =
          trainData(i)._1(j) - means(j)

        sum +=
          difference * difference
      }

      stds(j) =
        math.sqrt(sum / trainData.length)

      if (stds(j) == 0.0) {
        stds(j) = 1.0
      }
    }

    // ----------------------------------------------------------
    // 5. STANDARDIZE TRAINING DATA
    // ----------------------------------------------------------

    val train =
      trainData.map {

        case (features, label) =>

          val scaled =
            DenseVector.zeros[Double](numFeatures)

          for (j <- 0 until numFeatures) {

            scaled(j) =
              (features(j) - means(j)) /
                stds(j)
          }

          (scaled, label)
      }

    // ----------------------------------------------------------
    // 6. LOGISTIC REGRESSION OBJECTIVE
    // ----------------------------------------------------------

    val objective =
      new DiffFunction[DenseVector[Double]] {

        override def calculate(
                                weights: DenseVector[Double]
                              ): (Double, DenseVector[Double]) = {

          var loss = 0.0

          val gradient =
            DenseVector.zeros[Double](
              numFeatures + 1
            )

          for (i <- train.indices) {

            val x =
              train(i)._1

            val y =
              train(i)._2

            // Linear equation

            var z =
              weights(0)

            for (j <- 0 until numFeatures) {

              z +=
                weights(j + 1) * x(j)
            }

            // Sigmoid

            val p =
              sigmoid(z)

            // Prevent log(0)

            val safeP =
              math.max(
                math.min(p, 1.0 - 1e-15),
                1e-15
              )

            // Logistic loss

            loss +=
              -y * math.log(safeP) -
                (1.0 - y) *
                  math.log(1.0 - safeP)

            // Error

            val error =
              p - y

            gradient(0) +=
              error

            for (j <- 0 until numFeatures) {

              gradient(j + 1) +=
                error * x(j)
            }
          }

          // Average loss

          loss /=
            train.length

          // Average gradient

          for (j <- 0 to numFeatures) {

            gradient(j) /=
              train.length
          }

          (loss, gradient)
        }
      }

    // ----------------------------------------------------------
    // 7. TRAIN MODEL
    // ----------------------------------------------------------

    val initialWeights =
      DenseVector.zeros[Double](
        numFeatures + 1
      )

    val optimizer =
      new LBFGS[DenseVector[Double]](
        maxIter = 200,
        m = 5
      )

    val weights =
      optimizer.minimize(
        objective,
        initialWeights
      )

    println("Logistic Regression completed.")
    println("Generating graph...")

    // ==========================================================
    // 8. PREPARE GRAPH DATA
    // ==========================================================

    // Class 0 = No Diabetes

    val class0 =
      data
        .filter(_._2 == 0.0)
        .map(_._1)

    // Class 1 = Diabetes

    val class1 =
      data
        .filter(_._2 == 1.0)
        .map(_._1)

    // ----------------------------------------------------------
    // Glucose = column 1
    // BMI     = column 5
    // ----------------------------------------------------------

    val glucose0 =
      DenseVector(
        class0
          .map(_(1))
          .toArray
      )

    val bmi0 =
      DenseVector(
        class0
          .map(_(5))
          .toArray
      )

    val glucose1 =
      DenseVector(
        class1
          .map(_(1))
          .toArray
      )

    val bmi1 =
      DenseVector(
        class1
          .map(_(5))
          .toArray
      )

    // ==========================================================
    // 9. CREATE GRAPH
    // ==========================================================

    val figure =
      Figure(
        "Logistic Regression - Diabetes Classification"
      )

    val plotArea =
      figure.subplot(0)

    // ----------------------------------------------------------
    // CLASS 0
    // ----------------------------------------------------------
    // Breeze 2.1.0 supports "." and "+"
    // ----------------------------------------------------------

    plotArea += plot(
      glucose0,
      bmi0,
      '.'
    )

    // ----------------------------------------------------------
    // CLASS 1
    // ----------------------------------------------------------

    plotArea += plot(
      glucose1,
      bmi1,
      '+'
    )

    // ==========================================================
    // 10. DECISION BOUNDARY
    // ==========================================================

    val glucose =
      DenseVector(
        (50 to 200 by 5)
          .map(_.toDouble)
          .toArray
      )

    val bmi =
      DenseVector(
        glucose.toArray.map { g =>

          // Standardize glucose

          val glucoseScaled =
            (g - means(1)) /
              stds(1)

          // Equation:
          //
          // w0 + wGlucose*x +
          // wBMI*y = 0
          //

          val value =
            weights(0) +
              weights(2) *
                glucoseScaled

          val bmiScaled =
            -value /
              weights(6)

          // Convert BMI back

          bmiScaled *
            stds(5) +
            means(5)
        }
      )

    // ----------------------------------------------------------
    // Plot decision boundary
    // ----------------------------------------------------------

    plotArea +=
      plot(
        glucose,
        bmi
      )

    // ==========================================================
    // 11. GRAPH LABELS
    // ==========================================================

    plotArea.xlabel =
      "Glucose"

    plotArea.ylabel =
      "BMI"

    plotArea.title =
      "Logistic Regression - Diabetes Classification"

    // ==========================================================
    // 12. SAVE GRAPH
    // ==========================================================

    figure.saveas(
      "logistic_regression.png"
    )

    println()
    println("==============================================")
    println("GRAPH GENERATED SUCCESSFULLY")
    println("==============================================")

    println()
    println("Graph details:")
    println(" .  = Class 0 (No Diabetes)")
    println(" +  = Class 1 (Diabetes)")
    println(" Line = Decision Boundary")

    println()
    println(
      "Saved as: logistic_regression.png"
    )
  }
}