import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class GeneralTest {

    public static void main(String[] args) {

        for(int i = 0; i < 10; i++) {
            System.out.println(new Random().nextDouble());
        }

//        System.out.println((0.8 * 574.5));
//        System.out.println((int)(0.8 * 574.5));

//        List<Integer> range = IntStream.range(0, 200).filter(x -> x % 10 == 0).boxed().collect(Collectors.toList());
//        System.out.println(range);
    }

}
