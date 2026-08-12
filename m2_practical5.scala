
import breeze.linalg.DenseVector
import breeze.stats.mean
import scala.io.Source

object m2_practical5 {

  def main(args: Array[String]): Unit = {

    // ============================================
    // 1. Read CSV File
    // ============================================

    val filePath = "Advertising.csv"

    val source = Source.fromFile(filePath)

    // Skip header
    val lines = source.getLines().drop(1).toList

    source.close()

    // ============================================
    // 2. Extract TV and Sales
    // ============================================

    val data = lines.map { line =>

      // Remove quotation marks
      val values = line
        .split(",")
        .map(_.trim.replace("\"", ""))

      // CSV format:
      // 0 = ID
      // 1 = TV
      // 2 = Radio
      // 3 = Newspaper
      // 4 = Sales

      val tv = values(1).toDouble
      val sales = values(4).toDouble

      (tv, sales)
    }

    // ============================================
    // 3. Create Breeze Vectors
    // ============================================

    val x = DenseVector(data.map(_._1).toArray)
    val y = DenseVector(data.map(_._2).toArray)

    println("==============================================")
    println("       LINEAR REGRESSION USING BREEZE")
    println("==============================================")

    println()
    println("Dataset: Advertising.csv")
    println("Number of records: " + x.length)

    // ============================================
    // 4. Display First 10 Records
    // ============================================

    println()
    println("----------------------------------------------")
    println("FIRST 10 RECORDS")
    println("----------------------------------------------")

    for (i <- 0 until math.min(10, x.length)) {

      println(
        f"TV = ${x(i)}%8.2f   Sales = ${y(i)}%8.2f"
      )
    }

    // ============================================
    // 5. Calculate Mean
    // ============================================

    val meanX = mean(x)
    val meanY = mean(y)

    // ============================================
    // 6. Calculate Slope
    //
    // m = Σ((x - meanX)(y - meanY))
    //     -------------------------
    //       Σ((x - meanX)^2)
    // ============================================

    var numerator = 0.0
    var denominator = 0.0

    for (i <- 0 until x.length) {

      numerator +=
        (x(i) - meanX) * (y(i) - meanY)

      denominator +=
        math.pow(x(i) - meanX, 2)
    }

    val slope = numerator / denominator

    // ============================================
    // 7. Calculate Intercept
    //
    // c = meanY - m * meanX
    // ============================================

    val intercept =
      meanY - slope * meanX

    // ============================================
    // 8. Display Regression Equation
    // ============================================

    println()
    println("----------------------------------------------")
    println("REGRESSION RESULTS")
    println("----------------------------------------------")

    println(f"Mean of TV    = $meanX%.4f")
    println(f"Mean of Sales = $meanY%.4f")

    println(f"Slope         = $slope%.4f")
    println(f"Intercept     = $intercept%.4f")

    println()
    println("Regression Equation:")

    println(
      f"Sales = $intercept%.4f + $slope%.4f * TV"
    )

    // ============================================
    // 9. Calculate Predicted Sales
    // ============================================

    val predictions =
      DenseVector(
        x.toArray.map { value =>
          intercept + slope * value
        }
      )

    // ============================================
    // 10. Display Predictions
    // ============================================

    println()
    println("----------------------------------------------")
    println("PREDICTIONS")
    println("----------------------------------------------")

    for (i <- 0 until math.min(10, x.length)) {

      println(
        f"TV = ${x(i)}%8.2f   " +
          f"Actual Sales = ${y(i)}%8.2f   " +
          f"Predicted Sales = ${predictions(i)}%8.2f"
      )
    }

    // ============================================
    // 11. Calculate R-Squared
    // ============================================

    var ssTotal = 0.0
    var ssResidual = 0.0

    for (i <- 0 until y.length) {

      ssTotal +=
        math.pow(y(i) - meanY, 2)

      ssResidual +=
        math.pow(y(i) - predictions(i), 2)
    }

    val rSquared =
      1.0 - (ssResidual / ssTotal)

    // ============================================
    // 12. Display Model Accuracy
    // ============================================

    println()
    println("----------------------------------------------")
    println("MODEL ACCURACY")
    println("----------------------------------------------")

    println(f"R-Squared = $rSquared%.4f")

    // ============================================
    // 13. Predict New Sales
    // ============================================

    val newTV = 100.0

    val predictedSales =
      intercept + slope * newTV

    println()
    println("----------------------------------------------")
    println("NEW PREDICTION")
    println("----------------------------------------------")

    println(f"TV Advertising = $newTV%.2f")
    println(f"Predicted Sales = $predictedSales%.2f")

    // ============================================
    // 14. End
    // ============================================

    println()
    println("==============================================")
    println("             PROGRAM COMPLETED")
    println("==============================================")
  }
}
