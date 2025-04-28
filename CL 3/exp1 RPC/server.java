import java.io.*;
import java.net.*;

class server {
    public static void main(String[] args) throws Exception {
       ServerSocket sersock = new ServerSocket(4000);

        System.out.println("Server ready");

        Socket sock = sersock.accept();

        OutputStream ostream = sock.getOutputStream();
        PrintWriter pwrite = new PrintWriter(ostream, true);

        InputStream istream = sock.getInputStream();
        BufferedReader receiveRead = new BufferedReader(new InputStreamReader(istream));

        String fun;
        int a, b, c;

        while (true) {
            fun = receiveRead.readLine();

            if (fun != null) {
                System.out.println("Operation: " + fun);

                a = Integer.parseInt(receiveRead.readLine());
                System.out.println("Parameter 1: " + a);

                b = Integer.parseInt(receiveRead.readLine());
                System.out.println("Parameter 2: " + b);

                switch (fun) {
                    case "add":
                        c = a + b;
                        pwrite.println("Addition = " + c);
                        break;
                    case "sub":
                        c = a - b;
                        pwrite.println("Subtraction = " + c);
                        break;
                    case "mul":
                        c = a * b;
                        pwrite.println("Multiplication = " + c);
                        break;
                    case "div":
                        if (b != 0) {
                            c = a / b;
                            pwrite.println("Division = " + c);
                        } else {
                            pwrite.println("Error: Division by zero");
                        }
                        break;
                    default:
                        pwrite.println("Invalid operation");
                }

                System.out.flush();
            }
        }
    }
}
