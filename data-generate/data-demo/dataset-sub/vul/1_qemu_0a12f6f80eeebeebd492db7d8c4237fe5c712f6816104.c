static VAR1 *FUN1(VAR2 *VAR3)
{
    VAR4 *VAR5 = VAR3->VAR6;
    VAR1 *VAR7 = FUN2(VAR1, 1);
    *VAR7 = (VAR1){
        .VAR8  = VAR9,
        .VAR10.VAR11.VAR12 = FUN2(VAR13, 1),
    };
    if (VAR5->VAR14 == 2) {
        *VAR7->VAR10.VAR11.VAR12 = (VAR13){
            .VAR15             = FUN3(""),
            .VAR16      = VAR5->VAR16,
        };
    } else if (VAR5->VAR14 == 3) {
        *VAR7->VAR10.VAR11.VAR12 = (VAR13){
            .VAR15             = FUN3(""),
            .VAR17     = VAR5->VAR18 &
                                  VAR19,
            .VAR20 = true,
            .VAR21            = VAR5->VAR22 &
                                  VAR23,
            .VAR24        = true,
            .VAR16      = VAR5->VAR16,
        };
    } else {
        
        assert(false);
    }
    return VAR7;
}