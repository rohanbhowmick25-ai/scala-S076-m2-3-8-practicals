import scala.io.Source
import scala.collection.mutable

object FrequencyDistribution {

  def main(args: Array[String]): Unit = {

    val fileName = "Titanic-Dataset.csv"

    val lines = Source.fromFile(fileName).getLines().drop(1) // Skip header

    val frequency = mutable.Map[String, Int]()

    for (line <- lines) {
      val cols = line.split(",")

      // Pclass is the 3rd column (index = 2)
      val pclass = cols(2).trim

      frequency(pclass) = frequency.getOrElse(pclass, 0) + 1
    }

    println("======================================")
    println(" Frequency Distribution of Pclass")
    println("======================================")

    val sortedFreq = frequency.toSeq.sortBy(_._1)

    var cumulative = 0

    println(f"${"Pclass"}%-10s ${"Frequency"}%-10s ${"Cumulative"}%-10s")
    println("--------------------------------------")

    for ((category, freq) <- sortedFreq) {
      cumulative += freq
      println(f"$category%-10s $freq%-10d $cumulative%-10d")
    }
  }
}