import scala.io.Source

object SortTop5Rows {

  case class Passenger(
                        PassengerId: String,
                        Survived: String,
                        Pclass: String,
                        Name: String,
                        Sex: String,
                        Age: Double
                      )

  def main(args: Array[String]): Unit = {

    val fileName = "Titanic-Dataset.csv"

    val data = Source.fromFile(fileName)
      .getLines()
      .drop(1)
      .flatMap { line =>
        val cols = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)", -1)

        try {
          if (cols.length >= 6 && cols(5).nonEmpty)
            Some(Passenger(
              cols(0),
              cols(1),
              cols(2),
              cols(3),
              cols(4),
              cols(5).toDouble
            ))
          else
            None
        } catch {
          case _: Exception => None
        }
      }.toList

    // Sort by Age (Descending)
    val sorted = data.sortBy(_.Age).reverse

    println("Top 5 Oldest Passengers")
    println("---------------------------------------------------------------")
    println(f"${"ID"}%-5s ${"Name"}%-35s ${"Age"}%-8s ${"Sex"}%-8s ${"Pclass"}")
    println("---------------------------------------------------------------")

    sorted.take(5).foreach { p =>
      println(f"${p.PassengerId}%-5s ${p.Name.take(34)}%-35s ${p.Age}%-8.1f ${p.Sex}%-8s ${p.Pclass}")
    }
  }
}