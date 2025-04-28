import java.io.*;
import java.net.*;

class server {
    public static void main(String[] args) throws Exception {
        ServerSocket sersock = new ServerSocket(4000);

        System.out.println("Server ready, waiting for client...");

        Socket sock = sersock.accept();

        OutputStream ostream = sock.getOutputStream();
        PrintWriter pwrite = new PrintWriter(ostream, true);

        InputStream istream = sock.getInputStream();
        BufferedReader receiveRead = new BufferedReader(new InputStreamReader(istream));

        String receiveMessage;
        int num;
        long fact;  // factorial can be a big number, so use long

        while (true) {
            receiveMessage = receiveRead.readLine();

            if (receiveMessage == null) {
                break;  // client disconnected
            }

            System.out.println("Received number: " + receiveMessage);

            try {
                num = Integer.parseInt(receiveMessage);
                fact = 1;
                for (int i = 1; i <= num; i++) {
                    fact *= i;
                }
                pwrite.println(fact);  // send result
            } catch (NumberFormatException e) {
                pwrite.println("Invalid input");
            }
        }

        sock.close();
        sersock.close();
        System.out.println("Server exiting...");
    }
}
