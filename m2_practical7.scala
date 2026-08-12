import breeze.linalg.DenseVector

object KnnExample {

  // Data point containing features and class label
  case class DataPoint(
                        features: DenseVector[Double],
                        label: String
                      )

  def main(args: Array[String]): Unit = {

    // =====================================================
    // 1. CREATE DATASET
    // =====================================================

    val dataset = Seq(
      DataPoint(DenseVector(1.0, 2.0), "A"),
      DataPoint(DenseVector(1.5, 2.5), "A"),
      DataPoint(DenseVector(5.0, 6.0), "B"),
      DataPoint(DenseVector(5.5, 6.5), "B")
    )

    println("==========================================")
    println("        K-NEAREST NEIGHBOR (1-NN)")
    println("==========================================")

    println()
    println("Training Data:")
    println("------------------------------------------")

    dataset.foreach { point =>
      println(
        s"Features: ${point.features}   Label: ${point.label}"
      )
    }

    // =====================================================
    // 2. NEW DATA POINT
    // =====================================================

    val newPoint =
      DenseVector(1.8, 2.3)

    println()
    println("New Data Point:")
    println("------------------------------------------")
    println(s"Features: $newPoint")

    // =====================================================
    // 3. FIND NEAREST NEIGHBOR
    // =====================================================

    var minDistance =
      Double.MaxValue

    var predictedLabel =
      ""

    var nearestPoint:
      DenseVector[Double] = null

    println()
    println("Euclidean Distances:")
    println("------------------------------------------")

    for (point <- dataset) {

      // Calculate Euclidean distance:
      //
      // distance = sqrt(
      //   (x1-x2)^2 +
      //   (y1-y2)^2
      // )

      val difference =
        newPoint - point.features

      val distance =
        math.sqrt(
          difference dot difference
        )

      println(
        f"Distance to ${point.label}: $distance%.4f"
      )

      // Check whether this is the nearest point

      if (distance < minDistance) {

        minDistance =
          distance

        predictedLabel =
          point.label

        nearestPoint =
          point.features
      }
    }

    // =====================================================
    // 4. DISPLAY RESULT
    // =====================================================

    println()
    println("==========================================")
    println("        CLASSIFICATION RESULT")
    println("==========================================")

    println(
      f"Nearest Distance : $minDistance%.4f"
    )

    println(
      s"Nearest Point    : $nearestPoint"
    )

    println(
      s"Predicted Label   : $predictedLabel"
    )

    println("==========================================")
  }
}