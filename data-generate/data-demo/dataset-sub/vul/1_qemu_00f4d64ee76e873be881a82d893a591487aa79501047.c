static void FUN1(void *VAR1)
{
    VAR2 *VAR3 = VAR1;
    struct kvm_clock_data VAR4;
    int VAR5;
    if (VAR3->VAR6) {
        return;
    }
    VAR5 = FUN2(VAR7, VAR8, &VAR4);
    if (VAR5 < 0) {
        fprintf(VAR9, "", strerror(VAR5));
        VAR4.VAR10 = 0;
    }
    VAR3->VAR10 = VAR4.VAR10;
    
    VAR3->VAR6 = !FUN3();
}