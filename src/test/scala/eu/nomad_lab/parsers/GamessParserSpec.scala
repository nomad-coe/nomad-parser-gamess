package eu.nomad_lab.parsers

import eu.nomad_lab.{ parsers, DefaultPythonInterpreter }
import org.scalacheck.Properties
import org.specs2.mutable.Specification
import org.{ json4s => jn }

object GamessParserSpec extends Specification {
  "GamessParserTest" >> {
    "test exam01 " >> {
      "test with json-events" >> {
        ParserRun.parse(GamessParser, "parsers/gamess/test/examples/gamessus/exam01.out", "json-events") must_== ParseResult.ParseSuccess
      }
      "test with json" >> {
        ParserRun.parse(GamessParser, "parsers/gamess/test/examples/gamessus/exam01.out", "json") must_== ParseResult.ParseSuccess
      }
    }
    "test exam02 " >> {
      "test with json-events" >> {
        ParserRun.parse(GamessParser, "parsers/gamess/test/examples/gamessus/exam02.out", "json-events") must_== ParseResult.ParseSuccess
      }
      "test with json" >> {
        ParserRun.parse(GamessParser, "parsers/gamess/test/examples/gamessus/exam02.out", "json") must_== ParseResult.ParseSuccess
      }
    }
  }
}
