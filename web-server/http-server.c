//#include "mdb.h"
//#include "mylist.h"

#include <stdio.h>      /* for printf() and fprintf() */
#include <sys/socket.h> /* for socket(), bind(), and connect() */
#include <arpa/inet.h>  /* for sockaddr_in and inet_ntoa() */
#include <stdlib.h>     /* for atoi() and exit() */
#include <string.h>     /* for memset() */
#include <unistd.h>     /* for close() */
#include <signal.h>     /* for signal() */
#include <netdb.h>
#include <sys/types.h>

#define MAXPENDING 5    /* Maximum outstanding connection requests */

#define KeyMax 5
#define BUF_SIZE 4096
#define LIL_BUF 1024

static void die(const char *message)
{
    perror(message);
    exit(1); 
}

static int CreateMdbSocket( const char *mdbHost, unsigned short MdbPort);
void HandleMdbLookup(char *requestURI, FILE *mdbFile, int MdbSock, int clntSocket);
static void sendStatus(int clntSocket, int statusCode, char *reasonPhrase);
void HandleTCPClient(int clntSocket, int MdbSock, FILE *MdbFile, char *client_ip,  const char *webRoot);


int main(int argc, char *argv[])
{
    int servSock;                    /* Socket descriptor for server */
    int clntSock;                    /* Socket descriptor for client */
    struct sockaddr_in ServAddr;     /* Local address */
    struct sockaddr_in ClntAddr;     /* Client address */
    unsigned int clntLen;            /* Length of client address data struct */

    if (signal(SIGPIPE, SIG_IGN) == SIG_ERR) 
	die("signal() failed");

    if (argc != 5)  
    {
        fprintf(stderr, "Usage:  %s <server_port> <web_root> <mdb-lookup-host> <mdb-lookup-port>\n", argv[0]);
        exit(1);
    }

    unsigned short ServPort = atoi(argv[1]);  /* 2nd arg:  local port */
    const char *root = argv[2]; 
    const char *mdbHost = argv[3];
    unsigned short MdbPort = atoi(argv[4]);

   /* Create MDB SERVER  SOCKET */

   int mdbSock = CreateMdbSocket(mdbHost, MdbPort);
   FILE *mdbFile = fdopen(mdbSock, "r");
   if(mdbFile == NULL)
       die("backend connection failed");
        
    /* Create socket for incoming connections */
    if ((servSock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
        die("socket() failed");

    /* Construct local address structure */
    memset(&ServAddr, 0, sizeof(ServAddr));   // Zero out structure
    ServAddr.sin_family = AF_INET;                // Internet address family
    ServAddr.sin_addr.s_addr = htonl(INADDR_ANY); // Any incoming interface
    ServAddr.sin_port = htons(ServPort);      // Local port

    /* Bind to the local address */
    if (bind(servSock, (struct sockaddr *)&ServAddr, 
		sizeof(ServAddr)) < 0)
        die("bind() failed");

    /* Mark the socket so it will listen for incoming connections */
    if (listen(servSock, MAXPENDING) < 0)
        die("listen() failed");

    for (;;) /* Run forever */
    {
        /* Set the size of the in-out parameter */
        clntLen = sizeof(ClntAddr);

        /* Wait for a client to connect */
        if ((clntSock = accept(servSock, (struct sockaddr *) &ClntAddr, 
                               &clntLen)) < 0)
            die("accept() failed"); //u cant recover from this

        /* clntSock is connected to a client! */

        char *client_ip = inet_ntoa(ClntAddr.sin_addr);

        fprintf(stderr, "\nconnection started from: %s\n", 
		client_ip);

        HandleTCPClient(clntSock, mdbSock, mdbFile, client_ip,  root);

        /* close the client connection after each request */

        if(clntSock)
            close(clntSock);
	
	fprintf(stderr, "connection terminated from: %s\n", 
		client_ip);
    }

    if(mdbFile){
        fclose(mdbFile);
        close(mdbSock);
    }

    if(mdbSock){
        fclose(mdbFile);
        close(mdbSock);
    }
    close(servSock);

}
static int CreateMdbSocket(const char *MdbHost, unsigned short MdbPort){
    int MdbSock;
    struct sockaddr_in mdbAddr;
    struct hostent *he;
    if((he = gethostbyname(MdbHost)) == NULL){
        die("gethostbyname failed");
    }

    char *serverIP;
    serverIP = inet_ntoa(*(struct in_addr *)he->h_addr);
    if((MdbSock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
        die("socket failed");

    // construct mdb server address 
    memset(&mdbAddr, 0, sizeof(mdbAddr));
    mdbAddr.sin_family = AF_INET;
    mdbAddr.sin_addr.s_addr = inet_addr(serverIP);
    mdbAddr.sin_port = htons(MdbPort);

    if(connect(MdbSock, (struct sockaddr *)&mdbAddr, sizeof(mdbAddr)) < 0)
        die("connect failed");

    return MdbSock;
} 

static void sendStatus(int clntSocket, int statusCode, char *reasonPhrase){

    char buf[LIL_BUF];
    sprintf(buf, "HTTP/1.0 %d %s\r\n", statusCode, reasonPhrase);
    strcat(buf, "\r\n");

    if(statusCode != 200){ //if something went wrong, send a lil html body
        char body[LIL_BUF];
        sprintf(body, 
                "<html><body>\n"
                "<h1>%d %s</h1>\n"
                "</body></html>\n",
                statusCode, reasonPhrase);
        strcat(buf, body);
    }
    send(clntSocket, buf, strlen(buf), 0);

}

void HandleMdbLookup(char *requestURI, FILE *mdbFile, int MdbSock, int clntSocket){

    sendStatus(clntSocket, 200, "OK");

    const char *form = 
        "<html><body>\n"
        "<h1>mdb-lookup</h1>\n"
        "<p>\n"
        "<form method=GET action=/mdb-lookup>\n"
        "lookup: <input  type=text name=key>\n"
        "<input type=submit>\n"
        "</form>\n"
        "<p>\n"
        ;

    if(send(clntSocket, form, strlen(form), 0) != strlen(form)){
        perror("mdb form send failed");
        goto func_end;
    }

    const char *keyURI = "/mdb-lookup?key=";
    if(strncmp(requestURI, keyURI, strlen(keyURI)) == 0){
        //we got ourselves a mdb request
        const char *key = requestURI + strlen(keyURI);
    
        fprintf(stderr, "looking up [%s]\n", key);
        send(MdbSock, key, strlen(key), 0);
        send(MdbSock, "\n", strlen("\n"), 0);

        char line[LIL_BUF];
        char *table_header = "<p><table border>";

        if(send(clntSocket, table_header, strlen(table_header), 0) != strlen(table_header))
            goto func_end;

        /* receive all entries from mdb-lookup-server */
        for(;;){
            if(fgets(line, sizeof(line), mdbFile) == NULL){
                if(ferror(mdbFile)){
                    perror("\nmdb-lookup connection failed u");
                }
                else
                    fprintf(stderr, "mdb-lookup-server connection terminated\n");
                goto func_end;
            }
            /* mdb-request is complete */
            if(strcmp(line, "\n") == 0){
                break;
            }

            char *table_row = "\n<tr><td>";
            if(send(clntSocket, table_row, strlen(table_row), 0) != strlen(table_row)){
                perror("send error");
                goto func_end;
            }

            if(send(clntSocket, line, strlen(line), 0) != strlen(line))
                goto func_end;
        }
        /*loop end */

        char *table_footer = "\n</table>\n";
        if(send(clntSocket, table_footer, strlen(table_footer), 0) != strlen(table_footer)){
            perror("table send failed");
            goto func_end;
        }
    }

    char *form_end = "</body></html>\n";
    if(send(clntSocket, form_end, strlen(form_end), 0) != strlen(form_end)){
        goto func_end;
    }
            
func_end:

    return;
}

void HandleTCPClient(int clntSocket, int MdbSock, FILE *mdbFile, char *client_ip, const char *webRoot)
{
   

    int statusCode;
    char *reasonPhrase;
    char requestLine[BUF_SIZE];
    char file_buf[BUF_SIZE];
    char filePath[strlen(webRoot) + 1];

    FILE *fp = NULL;
    FILE *client = fdopen(clntSocket, "rb"); 

    if (client == NULL) {
	die("fdopen failed");
        goto tcp_end;
    }

    
    if(fgets(requestLine, sizeof(requestLine), client) == NULL){
       statusCode = 400; //client closed socket prematurely
       reasonPhrase = "Bad Request";
       goto tcp_end;
    }

    char *token_separators =  "\t \r\n"; //tab, space, nefleewline
    char *method = strtok(requestLine, token_separators);
    char *requestURI = strtok(NULL, token_separators);
    char *httpVersion = strtok(NULL, token_separators);
    
    if(!method || !requestURI || !httpVersion){
        statusCode = 501; reasonPhrase = "Not Implemented";
        sendStatus(clntSocket, statusCode, reasonPhrase);
        goto tcp_end;
    }
    if(strcmp(method, "GET") != 0){
        statusCode = 501; reasonPhrase = "Not Implemented";
        sendStatus(clntSocket, statusCode, reasonPhrase);
        goto tcp_end;
    }

    char request[256];
    if(requestURI)
        strcpy(request, requestURI);
    char methodCopy[sizeof(method)+1];
    if(method)
        strcpy(methodCopy, method);
    char http[sizeof(httpVersion)+1];
    if(httpVersion)
        strcpy(http, httpVersion);

    strcpy(filePath, webRoot);

    filePath[strlen(webRoot) + 1] = '\0';

    /* Append index.html if request URI ends with '\' */
    strcat(filePath, requestURI);
    if (filePath[(strlen(filePath)-1)] == '/'){
        strcat(filePath, "index.html");
    }


        /* only support HTTP/1.0 & 1.1 */
    if (strcmp("HTTP/1.0", httpVersion) != 0 && strcmp("HTTP/1.1", httpVersion) != 0){

        statusCode = 501; reasonPhrase = "Not Implemented";
        sendStatus(clntSocket, statusCode, reasonPhrase);
        goto tcp_end;
    }

        /* Check that request URI starts with " / " */
    if(*request != '/'|| strstr(request, "/../") || strstr(request, "/..")) {

        statusCode = 400; reasonPhrase = "Bad Request";
        sendStatus(clntSocket, statusCode, reasonPhrase);
        goto tcp_end;
    }

        /* LOOP TO MOVE PAST HEADER LINES */

    for(;;){
        if(fgets(requestLine, sizeof(requestLine), client) == NULL){ //POTENTIAL ERROR SOURCE??

            statusCode = 400;
            reasonPhrase = "Bad Request";
           // sendStatus(clntSocket, statusCode, reasonPhrase);
            goto tcp_end;

        }
        if(strcmp("\r\n", requestLine) == 0 || strcmp("\n", requestLine) == 0){
            fprintf(stderr, "moved past headers\n");
            break;
        }        
    }
        /* NOW U ARE PAST HEADERS */


    char *mdbCase1 = "/mdb-lookup";
    char *mdbCase2 = "/mdb-lookup?";

    if(strcmp(mdbCase1, request) == 0 || strncmp(mdbCase2, request, strlen(mdbCase2)) == 0){

            statusCode = 200; reasonPhrase = "OK";
            HandleMdbLookup(request, mdbFile, MdbSock, clntSocket);
    }

        /* IF NOT MDB, BEGIN HANDLING A FILE REQUEST */           

    else{
        
        fp = fopen(filePath, "rb"); /* open file specified by webRoot*/
        if (fp == NULL) {
            /* client closed connection prematurely */

            statusCode = 404; reasonPhrase = "File Not Found";
            sendStatus(clntSocket, statusCode, reasonPhrase);
            
            goto tcp_end;
        }
        
        /* otherwise SEND 200 OK status!!! */ 

        statusCode = 200; reasonPhrase = "OK";
        sendStatus(clntSocket, statusCode, reasonPhrase);

        size_t n;
        while ((n = fread(file_buf, 1, sizeof(file_buf), fp)) > 0)
        {
                if(send(clntSocket, file_buf, n, 0) != n) {
                    perror("\nsend failed\n");
                    break;
                }
        }
        if(ferror(fp))
            perror("fread() failed\n");
    }
    
tcp_end:

    fprintf(stderr, "%s \"%s %s %s\" %d %s\n", client_ip, methodCopy, request, http, statusCode, reasonPhrase);

    if(fp)
        fclose(fp);
    if(client)
        fclose(client);
}

