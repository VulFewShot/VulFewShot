void FUN1(bool VAR1, int64_t VAR2, VAR3 **VAR4)
{
    int VAR5;
    int VAR6;
    pid_t VAR7;
    VAR3 *VAR8 = NULL;
    struct timeval VAR9;
    
    if (VAR1) {
        
        if (VAR2 / 1000000000 != (VAR10)(VAR2 / 1000000000)) {
            FUN2(VAR4, "" VAR11 "", VAR2);
            return;
        }
        VAR9.VAR12 = VAR2 / 1000000000;
        VAR9.VAR13 = (VAR2 % 1000000000) / 1000;
        FUN3(&VAR14, VAR9.VAR12);
        if (VAR14.VAR15 < 1970 || VAR14.VAR15 >= 2070) {
            FUN4(VAR4, VAR16, "");
            return;
        }
        VAR5 = FUN5(&VAR9, NULL);
        if (VAR5 < 0) {
            FUN4(VAR4, VAR16, "");
            return;
        }
    }
    
    VAR7 = FUN6();
    if (VAR7 == 0) {
        FUN7();
        FUN8(0);
        FUN8(1);
        FUN8(2);
        
        execle("", "", VAR1 ? "" : "",
               NULL, VAR17);
        FUN9(VAR18);
    } else if (VAR7 < 0) {
        FUN4(VAR4, VAR16, "");
        return;
    }
    FUN10(VAR7, &VAR6, &VAR8);
    if (VAR8) {
        FUN11(VAR4, VAR8);
        return;
    }
    if (!FUN12(VAR6)) {
        FUN2(VAR4, "");
        return;
    }
    if (FUN13(VAR6)) {
        FUN2(VAR4, "");
        return;
    }
}