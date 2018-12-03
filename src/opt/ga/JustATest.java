package opt.ga;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class JustATest {


   public static void main(String[] args) {

       System.out.println("JustATest.main");

//       int number_of_records = 500;
//       int number_of_dimensions = 100;
//       int total_number_of_bits = number_of_records * number_of_dimensions;
//       double mutationRate = 0.001;
//
//       int total_number_of_bits_to_mutate = (int) Math.round(new Double(total_number_of_bits) * mutationRate);
//
//       System.out.println("total_number_of_bits_to_mutate: " + total_number_of_bits_to_mutate);
//
//       List<Integer> range = IntStream.range(0, total_number_of_bits).boxed().collect(Collectors.toList());
//       Collections.shuffle(range);
//       List<Integer> bits_to_mutate = range.subList(0, total_number_of_bits_to_mutate);
//       Collections.sort(bits_to_mutate);
//       System.out.println("bits_to_mutate: " + bits_to_mutate);
//       for(int index = 0; index < bits_to_mutate.size(); index++) {
//
//            int instance_to_mutate = bits_to_mutate.get(index)/number_of_dimensions;
//            int this_instance_bit_to_mutate = bits_to_mutate.get(index) % number_of_dimensions;
//            System.out.println("instance_to_mutate and bit: " + instance_to_mutate + " " + this_instance_bit_to_mutate);
//
//       }

       System.out.println(1 % 2);
       System.out.println(2 % 2);





   }
}
