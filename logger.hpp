#include <iostream>
#include <string>


/**Barebones logging facility.*/


#define SPAM 0
#define DEBUG 1 
#define SERIOUS 2
#ifdef ENABLE_LOGGING

    inline void Log(int level, std::string msg)
    {
        static const int minlevel = SPAM;
        if(level < minlevel)
        {
            return;
        }
        
        std::cout<<"LOG; level: " << level << "; "  << msg << "\n";
    }
#endif


#ifndef ENABLE_LOGGING
    #define Log(level, message) ; //lol.
#endif 


//concise 'throw if' - used for checking external problems. Allows the user code to recover with standard try/catch
void check(bool in)
{
    if(!in)
    {
        throw std::runtime_error("User assert failed!");
    }
}
