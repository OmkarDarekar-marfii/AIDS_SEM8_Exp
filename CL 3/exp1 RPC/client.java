import java.io.*;
import java.net.*;

class client {
    public static void main(String[] args) throws Exception {
        Socket sock = new Socket("127.0.0.1", 4000);

        BufferedReader keyRead = new BufferedReader(new InputStreamReader(System.in));

        OutputStream ostream = sock.getOutputStream();
        PrintWriter pwrite = new PrintWriter(ostream, true);

        InputStream istream = sock.getInputStream();
        BufferedReader receiveRead = new BufferedReader(new InputStreamReader(istream));

        System.out.println("Client ready, type number to calculate factorial");

        String receiveMessage, sendMessage;

        while (true) {
            System.out.print("\nEnter an integer (or 'exit' to quit): ");
            sendMessage = keyRead.readLine();

            if (sendMessage.equalsIgnoreCase("exit")) {
                sock.close();
                System.out.println("Client exiting...");
                break;
            }

            pwrite.println(sendMessage);  // send number to server

            if ((receiveMessage = receiveRead.readLine()) != null) {  // read factorial result
                System.out.println("Factorial from Server: " + receiveMessage);
            }
        }
    }
}
