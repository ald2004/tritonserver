 #include "tritonserver.h"
//#include "http_server.h"


bool Parse(TRITONSERVER_ServerOptions** server_options, int argc, char** argv);
int main(int argc,char** argv)
{
    
    
    TRITONSERVER_ServerOptions* server_options = nullptr;
    if (!Parse(&server_options, argc, argv)) {
        exit(1);
    }
    std::cout << server_options << std::endl;
    return 0;
}