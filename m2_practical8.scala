import breeze.linalg._
import breeze.plot._

import scala.io.Source

object KMeansExample {

  def main(args: Array[String]): Unit = {

    // ==========================================================
    // K-MEANS CLUSTERING USING BREEZE
    // Dataset: Mall Customers
    // ==========================================================

    println("==============================================")
    println("       K-MEANS CUSTOMER SEGMENTATION")
    println("==============================================")

    // ----------------------------------------------------------
    // 1. READ CSV FILE
    // ----------------------------------------------------------

    val filePath = "Mall_Customers.csv"

    val source = Source.fromFile(filePath)

    val lines =
      source
        .getLines()
        .drop(1)
        .toArray

    source.close()

    println()
    println("Dataset loaded successfully.")
    println("Total records: " + lines.length)

    // ----------------------------------------------------------
    // 2. EXTRACT FEATURES
    // ----------------------------------------------------------
    //
    // CSV columns:
    //
    // 0 = CustomerID
    // 1 = Gender
    // 2 = Age
    // 3 = Annual Income (k$)
    // 4 = Spending Score (1-100)
    //
    // We use only:
    //
    // Annual Income
    // Spending Score
    //
    // ----------------------------------------------------------

    val dataArray =
      lines.map { line =>

        val values =
          line
            .split(",")
            .map(_.trim.replace("\"", ""))

        Array(
          values(3).toDouble,
          values(4).toDouble
        )
      }

    val data =
      DenseMatrix(dataArray: _*)

    println(
      "Number of samples: " + data.rows
    )

    println(
      "Number of features: " + data.cols
    )

    // ----------------------------------------------------------
    // 3. K-MEANS PARAMETERS
    // ----------------------------------------------------------

    val k = 5

    val maxIterations = 100

    println()
    println("Number of clusters K: " + k)

    // ----------------------------------------------------------
    // 4. INITIALIZE CENTROIDS
    // ----------------------------------------------------------

    var centroids =
      DenseMatrix.zeros[Double](
        k,
        data.cols
      )

    val randomIndices =
      scala.util.Random
        .shuffle(
          (0 until data.rows).toList
        )
        .take(k)

    // Copy each value individually
    for (i <- 0 until k) {

      for (j <- 0 until data.cols) {

        centroids(i, j) =
          data(randomIndices(i), j)
      }
    }
    println()
    println("----------------------------------------------")
    println("INITIAL CENTROIDS")
    println("----------------------------------------------")

    println(centroids)

    // ----------------------------------------------------------
    // 5. INITIALIZE ASSIGNMENTS
    // ----------------------------------------------------------

    var assignments =
      DenseVector.zeros[Int](
        data.rows
      )

    var previousAssignments =
      DenseVector.fill[Int](
        data.rows
      )(-1)

    var iteration = 0

    var converged = false

    // ==========================================================
    // 6. K-MEANS ALGORITHM
    // ==========================================================

    while (
      iteration < maxIterations &&
        !converged
    ) {

      println()
      println(
        "--- Iteration " +
          (iteration + 1) +
          " ---"
      )

      // --------------------------------------------------------
      // ASSIGNMENT STEP
      // --------------------------------------------------------

      for (i <- 0 until data.rows) {

        val point =
          data(i, ::).t

        var minimumDistance =
          Double.MaxValue

        var closestCluster =
          0

        for (j <- 0 until k) {

          val centroid =
            centroids(j, ::).t

          // Euclidean distance

          val difference =
            point - centroid

          val distance =
            math.sqrt(
              difference dot difference
            )

          if (
            distance < minimumDistance
          ) {

            minimumDistance =
              distance

            closestCluster =
              j
          }
        }

        assignments(i) =
          closestCluster
      }

      // --------------------------------------------------------
      // CHECK CONVERGENCE
      // --------------------------------------------------------

      if (
        assignments == previousAssignments
      ) {

        converged = true

        println(
          "Clusters converged."
        )

      } else {

        previousAssignments =
          assignments.copy
      }

      // --------------------------------------------------------
      // UPDATE CENTROIDS
      // --------------------------------------------------------

      val newCentroids =
        DenseMatrix.zeros[Double](
          k,
          data.cols
        )

      val clusterCounts =
        DenseVector.zeros[Int](k)

      for (i <- 0 until data.rows) {

        val cluster =
          assignments(i)

        for (j <- 0 until data.cols) {

          newCentroids(
            cluster,
            j
          ) += data(i, j)
        }

        clusterCounts(cluster) += 1
      }

      // --------------------------------------------------------
      // CALCULATE MEAN FOR EACH CLUSTER
      // --------------------------------------------------------

      for (i <- 0 until k) {

        if (
          clusterCounts(i) > 0
        ) {

          for (j <- 0 until data.cols) {

            newCentroids(i, j) =
              newCentroids(i, j) /
                clusterCounts(i).toDouble
          }
        }
      }

      centroids =
        newCentroids

      println(
        "Cluster sizes: " +
          (0 until k)
            .map(
              i => clusterCounts(i)
            )
            .mkString(", ")
      )

      iteration += 1
    }

    // ==========================================================
    // 7. FINAL RESULTS
    // ==========================================================

    println()
    println("==============================================")
    println("              FINAL RESULTS")
    println("==============================================")

    println()
    println(
      "K-Means converged in " +
        iteration +
        " iterations."
    )

    println()
    println("Final Centroids:")
    println("----------------------------------------------")

    for (i <- 0 until k) {

      println(
        "Cluster " +
          i +
          ": " +
          centroids(i, ::)
      )
    }

    // ==========================================================
    // 8. DISPLAY SOME CLUSTER ASSIGNMENTS
    // ==========================================================

    println()
    println("Sample Cluster Assignments:")
    println("----------------------------------------------")

    for (i <- 0 until math.min(20, data.rows)) {

      println(
        "Customer " +
          (i + 1) +
          " -> Cluster " +
          assignments(i)
      )
    }

    // ==========================================================
    // 9. PREPARE GRAPH DATA
    // ==========================================================

    val clusterData =
      (0 until k).map { clusterId =>

        val indices =
          (0 until data.rows)
            .filter(
              i =>
                assignments(i) == clusterId
            )

        val x =
          DenseVector(
            indices
              .map(
                i => data(i, 0)
              )
              .toArray
          )

        val y =
          DenseVector(
            indices
              .map(
                i => data(i, 1)
              )
              .toArray
          )

        (x, y)
      }

    // ==========================================================
    // 10. CREATE GRAPH
    // ==========================================================

    println()
    println("Generating graph...")

    val figure =
      Figure(
        "K-Means Customer Segmentation"
      )

    val plotArea =
      figure.subplot(0)

    // Plot each cluster

    for (i <- 0 until k) {

      val x =
        clusterData(i)._1

      val y =
        clusterData(i)._2

      plotArea += plot(
        x,
        y,
        '.'
      )
    }

    // ==========================================================
    // 11. PLOT CENTROIDS
    // ==========================================================

    val centroidX =
      DenseVector(
        (0 until k)
          .map(
            i => centroids(i, 0)
          )
          .toArray
      )

    val centroidY =
      DenseVector(
        (0 until k)
          .map(
            i => centroids(i, 1)
          )
          .toArray
      )

    plotArea += plot(
      centroidX,
      centroidY,
      '+'
    )

    // ==========================================================
    // 12. GRAPH LABELS
    // ==========================================================

    plotArea.xlabel =
      "Annual Income (k$)"

    plotArea.ylabel =
      "Spending Score (1-100)"

    plotArea.title =
      "K-Means Customer Segmentation"

    // ==========================================================
    // 13. SAVE GRAPH
    // ==========================================================

    figure.saveas(
      "kmeans_mall_customers.png"
    )

    println()
    println("==============================================")
    println("       GRAPH GENERATED SUCCESSFULLY")
    println("==============================================")

    println()
    println(
      "Saved as: kmeans_mall_customers.png"
    )

    println()
    println("X-axis  : Annual Income")
    println("Y-axis  : Spending Score")
    println("Clusters: " + k)
    println("'.'     : Customer")
    println("'+'     : Cluster Centroid")
  }
}